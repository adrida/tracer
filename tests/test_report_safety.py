"""Untrusted trace text must remain text in reports, including script data."""
import json
from dataclasses import asdict
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tracer.analysis.html_report import generate_html_report
from tracer.scanner import ScanResult, scan_html
from tracer.types import ArtifactManifest

PAYLOAD = '</script><script>window.TRACER_REPORT_TEST=1</script><img src=x onerror="alert(1)">'


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.scripts = []
        self.in_script = False
    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, attrs))
        if tag == 'script':
            self.in_script = True
            self.scripts.append('')
    def handle_endtag(self, tag):
        if tag == 'script': self.in_script = False
    def handle_data(self, text):
        if self.in_script: self.scripts[-1] += text


def assert_inert(html):
    parsed = Elements(); parsed.feed(html)
    assert 'window.TRACER_REPORT_TEST=1' not in parsed.scripts
    assert not any(tag == 'img' for tag, attrs in parsed.tags)
    assert not any(key.lower().startswith('on') for tag, attrs in parsed.tags for key, value in attrs)


def test_scan_projection_cannot_end_script_element():
    projection = {'points': [[0, 0, 0, 0]], 'clusters': {'0': {'label': PAYLOAD, 'ex': [PAYLOAD]}}}
    result = ScanResult(1, 1, 1, .9, 0, 0, projection=projection)
    html = scan_html(result)
    assert_inert(html)
    assert '\\u003c/script' in html


@pytest.fixture
def report_files(tmp_path):
    manifest = asdict(ArtifactManifest(version='0.1.0', n_traces=20, label_space=[PAYLOAD],
                                       selected_method=PAYLOAD, embedding_dim=PAYLOAD,
                                       coverage_cal=.5, teacher_agreement_cal=1))
    report = {
        'coverage': .5,
        'slices': [{'slice_name': 'label:' + PAYLOAD, 'count': 20, 'handled_rate': .5,
                    'deferred_rate': .5, 'teacher_agreement_handled': 1}],
        'boundary_pairs': [{'teacher_label': PAYLOAD, 'handled_preview': PAYLOAD,
                            'deferred_preview': PAYLOAD, 'handled_score': .9, 'deferred_score': .7}],
        'handled_examples': [{'input_preview': PAYLOAD, 'teacher_label': PAYLOAD, 'accept_score': .9}],
        'deferred_examples': [{'input_preview': PAYLOAD, 'teacher_label': PAYLOAD}],
        'temporal_deltas': [{'label': PAYLOAD, 'previous_handled_rate': .4, 'current_handled_rate': .5, 'delta': .1}],
    }
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'qualitative_report.json').write_text(json.dumps(report))
    return tmp_path, manifest, report


def test_fit_report_escapes_trace_and_manifest_fields(report_files):
    root, _, _ = report_files
    html = Path(generate_html_report(root)).read_text()
    assert_inert(html)
    assert '&lt;' in html


def test_sankey_labels_and_title_are_plain_text(report_files):
    pytest.importorskip('plotly')
    from tracer.analysis.sankey import generate_sankey, _build_sankey_figure
    root, manifest, report = report_files
    fig = _build_sankey_figure(manifest, report, 15, False)
    assert all('<' not in label for label in fig.data[0].node.label)
    html = Path(generate_sankey(root, title=PAYLOAD)).read_text()
    assert_inert(html)
