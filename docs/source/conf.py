import os
import sys
import inspect

sys.path.insert(0, os.path.abspath("../../"))

project = "mTopic"
copyright = "2025, Piotr Rutkowski"
author = "Piotr Rutkowski"
release = "1.0"

extensions = [
    "recommonmark",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx_automodapi.automodapi",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

# autodoc/autosummary config
autosummary_generate = True
autosummary_imported_members = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_book_theme"

source_suffix = {
    ".rst": "restructuredtext",
}

html_logo = "_static/mTopic_logo_light.png"

html_theme_options = {
    "navigation_depth": 4,
    "logo": {
        "image_light": "_static/mTopic_logo_light.png",
        "image_dark": "_static/mTopic_logo_dark.png",
    },
}

html_static_path = ["_static"]
html_extra_path = ["_static/mTopic_logo_dark.png", "_static/mTopic_logo_light.png"]
html_css_files = ["custom.css"]

from sphinx.ext.autosummary import Autosummary
from docutils.parsers.rst import directives


class AutoAutoSummary(Autosummary):
    option_spec = {"methods": directives.unchanged, "attributes": directives.unchanged}
    required_arguments = 1

    @staticmethod
    def get_members(obj, typ, include_public=None):
        include_public = include_public or []
        items = []

        for name, member in inspect.getmembers(obj):
            if typ == "method":
                if inspect.isroutine(member):  # functions, methods, builtins, descriptors
                    items.append(name)
            elif typ == "attribute":
                if (not inspect.isroutine(member)) and (not inspect.isclass(member)) and (not inspect.ismodule(member)):
                    items.append(name)

        public = [x for x in items if x in include_public or not x.startswith("_")]
        return public, items

    def run(self):
        clazz = str(self.arguments[0])
        try:
            module_name, class_name = clazz.rsplit(".", 1)
            m = __import__(module_name, globals(), locals(), [class_name])
            c = getattr(m, class_name)

            if "methods" in self.options:
                _, methods = self.get_members(c, "method", ["__init__"])
                self.content = [f"~{clazz}.{method}" for method in methods if not method.startswith("_")]

            if "attributes" in self.options:
                _, attribs = self.get_members(c, "attribute")
                self.content = [f"~{clazz}.{attrib}" for attrib in attribs if not attrib.startswith("_")]
        finally:
            return super().run()


def setup(app):
    app.add_directive("autoautosummary", AutoAutoSummary)
