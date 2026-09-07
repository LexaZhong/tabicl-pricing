"""Introspect the installed tabicl API so the wrappers target the real signatures."""

import inspect

import tabicl

print("tabicl version:", getattr(tabicl, "__version__", "?"))
print("exports:", [n for n in dir(tabicl) if not n.startswith("_")])
print()

for name in [
    "TabICLClassifier",
    "TabICLRegressor",
    "FinetunedTabICLClassifier",
    "FinetunedTabICLRegressor",
]:
    cls = getattr(tabicl, name, None)
    if cls is None:
        print(f"--- {name}: NOT AVAILABLE ---\n")
        continue
    print(f"--- {name} ---")
    try:
        sig = inspect.signature(cls.__init__)
        for pname, p in list(sig.parameters.items()):
            if pname == "self":
                continue
            print(f"  {pname} = {p.default!r}")
    except (ValueError, TypeError) as exc:
        print("  <signature unavailable>", exc)
    for meth in ["fit", "predict", "predict_proba"]:
        fn = getattr(cls, meth, None)
        if fn is not None:
            try:
                print(f"  .{meth}{inspect.signature(fn)}")
            except (ValueError, TypeError):
                print(f"  .{meth}(?)")
    print()
