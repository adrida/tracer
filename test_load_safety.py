import sys
from tracer.runtime.router import Router

print("Testing runtime initialization with null selected_method...")

try:
    # Attempt to load from our failed .tracer directory
    router = Router.load(".tracer")
    print(" BUG STILL PRESENT: The router successfully loaded a stale pipeline artifact!")
    sys.exit(1)
except ValueError as e:
    print("\n SUCCESS! Your patch successfully caught the stale pipeline error:")
    print(f"Captured Error message: {e}")
    sys.exit(0)
except Exception as e:
    print(f" UNEXPECTED CRASH: {type(e).__name__}: {e}")
    sys.exit(1)