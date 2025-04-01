#! python 3
# flag: python.reloadEngine

import scriptcontext as sc

from Rhino import RhinoApp
from Rhino.UI import RhinoEtoApp, EtoExtensions
from ui.Views import RenderView

dialog = RenderView()

parent = RhinoEtoApp.MainWindowForDocument(sc.doc)
EtoExtensions.ShowSemiModal(dialog, sc.doc, parent)
