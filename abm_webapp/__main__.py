"""
Allow ``python -m abm_webapp`` to start the webapp.
"""
from abm_webapp.app import main

if __name__ == "__main__":
    raise SystemExit(main())
