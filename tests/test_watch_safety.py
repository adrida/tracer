"""Tracing must observe async work and remain harmless when storage fails."""
import asyncio
import inspect

import pytest

from tracer.watch import GenAISpan, LocalFileSink, MultiSink, Watcher


class Capture:
    def __init__(self): self.spans = []
    def emit(self, span): self.spans.append(span.to_dict())
    def close(self): pass


class Broken:
    def emit(self, span): raise OSError('recording unavailable')
    def close(self): raise OSError('close unavailable')


def test_async_results_errors_and_concurrent_parentage():
    captured = Capture()
    watcher = Watcher('async', sink=captured)

    @watcher
    async def child(text):
        await asyncio.sleep(.01)
        return text.upper()

    @watcher
    async def parent(text):
        return await child(text)

    @watcher
    async def failed(text):
        await asyncio.sleep(0)
        raise ValueError('application error')

    async def run():
        assert await asyncio.gather(parent('a'), parent('b')) == ['A', 'B']
        with pytest.raises(ValueError, match='application error'):
            await failed('failure')

    assert inspect.iscoroutinefunction(parent)
    asyncio.run(run())
    roots = [s for s in captured.spans if s['parent_span_id'] is None and s['status'] == 'ok']
    children = [s for s in captured.spans if s['parent_span_id'] is not None]
    assert len(roots) == len(children) == 2
    assert roots[0]['trace_id'] != roots[1]['trace_id']
    for child_span in children:
        parent_span = next(s for s in roots if s['span_id'] == child_span['parent_span_id'])
        assert child_span['trace_id'] == parent_span['trace_id']
        assert child_span['output_text'] == child_span['input_text'].upper()
        assert child_span['latency_ms'] >= 5
    assert captured.spans[-1]['status'] == 'error'
    assert captured.spans[-1]['error'] == 'application error'


def test_async_cancellation_is_recorded_and_reraised():
    captured = Capture()
    watcher = Watcher('cancel', sink=captured)
    @watcher
    async def cancelled():
        raise asyncio.CancelledError()
    async def run():
        with pytest.raises(asyncio.CancelledError):
            await cancelled()
    asyncio.run(run())
    assert captured.spans[0]['status'] == 'error'


@pytest.mark.parametrize('sink_type', ['local', 'custom', 'fanout'])
def test_sink_failure_preserves_results_and_original_errors(tmp_path, sink_type):
    captured = Capture()
    if sink_type == 'local':
        sink = LocalFileSink('blocked', str(tmp_path))
        (tmp_path / 'blocked.jsonl').mkdir()
    elif sink_type == 'fanout':
        sink = MultiSink([Broken(), captured])
    else:
        sink = Broken()
    watcher = Watcher('safe', sink=sink)
    assert watcher(lambda: 'success')() == 'success'
    original = ValueError('original failure')
    def fail(): raise original
    with pytest.raises(ValueError) as raised:
        watcher(fail)()
    assert raised.value is original
    watcher.close()
    if sink_type == 'fanout':
        assert [s['status'] for s in captured.spans] == ['ok', 'error']


def test_extractors_cannot_break_the_application():
    def broken(*args): raise TypeError('bad extractor')
    sink = Capture()
    watcher = Watcher('extract', sink=sink, extract_input=broken, extract_output=broken)
    assert watcher(lambda value: value)(object()) is not None
    assert sink.spans[0]['status'] == 'ok'


def test_debug_output_failure_cannot_break_the_application(monkeypatch):
    import io
    import sys

    closed_stderr = io.StringIO()
    closed_stderr.close()
    monkeypatch.setenv('TRACER_WATCH_DEBUG', '1')
    monkeypatch.setattr(sys, 'stderr', closed_stderr)
    watcher = Watcher('debug', sink=Broken())
    assert watcher(lambda: 'success')() == 'success'
    watcher.close()


@pytest.mark.parametrize('name', ['../escape', '/absolute', r'..\escape', '.', '..', '',
                                  'line\nbreak', 'trailing\n', 'nul\0byte', 'x' * 129])
def test_watcher_filename_rejects_unsafe_names(tmp_path, name):
    with pytest.raises(ValueError, match='Watcher names'):
        LocalFileSink(name, str(tmp_path / 'watch'))
    assert not (tmp_path / 'escape.jsonl').exists()


def test_safe_watcher_filename_stays_in_directory(tmp_path):
    sink = LocalFileSink('service-1.production_trace', str(tmp_path))
    sink.emit(GenAISpan(input_text='ok'))
    assert (tmp_path / 'service-1.production_trace.jsonl').is_file()
