import os
import site
import sys
from distutils.sysconfig import get_python_lib

from setuptools import setup

# Allow editable install into user site directory.

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# Warn if we are installing over top of an existing installation. 
overlay_warning = False
if "install" in sys.argv:
    lib_paths = [get_python_lib()]
    if lib_paths[0].startswith("/usr/lib/"):

        lib_paths.append(get_python_lib(prefix="/usr/local"))
    for lib_path in lib_paths:
        existing_path = os.path.abspath(os.path.join(lib_path, "django"))
        if os.path.exists(existing_path):

            overlay_warning = True
            break


setup()


if overlay_warning:
    sys.stderr.write(
        """
========
WARNING!
========
You have just installed Django over top of an existing
installation, without removing it first. 
"""
 
    )