import os
import sys
import inspect

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
project = "mTopic"
copyright = "2026, Piotr Rutkowski"
author = "Piotr Rutkowski"
release = "1.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx_automodapi.automodapi",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False

# -- Autosectionlabel --------------------------------------------------------
# Prefix labels with document name to avoid duplicate-label warnings
autosectionlabel_prefix_document = True

# -- Napoleon (Google/NumPy style docstrings) --------------------------------
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

# -- nbsphinx ----------------------------------------------------------------
# "never": use stored notebook outputs (recommended for reproducible builds)
# "auto":  execute notebooks without stored outputs
# "always": always re-execute
nbsphinx_execute = "never"

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_book_theme"

html_theme_options = {
    "navigation_depth": 4,
    "logo": {
        "image_light": "_static/mTopic_logo_light.png",
        "image_dark": "_static/mTopic_logo_dark.png",
    },
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]


# -- Custom directive: AutoAutoSummary ---------------------------------------
from sphinx.ext.autosummary import Autosummary
from docutils.parsers.rst import directives


class AutoAutoSummary(Autosummary):
    """Autosummary that automatically lists methods/attributes of a class."""

    option_spec = {
        "methods": directives.unchanged,
        "attributes": directives.unchanged,
    }
    required_arguments = 1

    @staticmethod
    def get_members(obj, typ, include_public=None):
        include_public = include_public or []
        items = []

        for name, member in inspect.getmembers(obj):
            if typ == "method":
                if inspect.isroutine(member):
                    items.append(name)
            elif typ == "attribute":
                if (
                    not inspect.isroutine(member)
                    and not inspect.isclass(member)
                    and not inspect.ismodule(member)
                ):
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
                self.content = [
                    f"~{clazz}.{method}"
                    for method in methods
                    if not method.startswith("_")
                ]

            if "attributes" in self.options:
                _, attribs = self.get_members(c, "attribute")
                self.content = [
                    f"~{clazz}.{attrib}"
                    for attrib in attribs
                    if not attrib.startswith("_")
                ]
        finally:
            return super().run()


def setup(app):
    app.add_directive("autoautosummary", AutoAutoSummary)
