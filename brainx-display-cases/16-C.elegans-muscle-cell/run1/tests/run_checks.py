"""Dependency-free focused check runner for environments without pytest."""

import tests.test_model as suite


def main():
    tests = [getattr(suite, name) for name in dir(suite) if name.startswith("test_")]
    for test in tests:
        print(f"RUN  {test.__name__}")
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} checks passed")


if __name__ == "__main__":
    main()
