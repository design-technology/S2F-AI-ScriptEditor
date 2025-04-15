import scriptcontext as sc
from random import randint

from Rhino.Display import RhinoView
from System.Drawing import Size

from Rhino import RhinoApp
from Rhino.UI import RhinoEtoApp, EtoExtensions

import Eto.Forms as ef
import Eto.Drawing as ed

class RenderView(ef.Dialog):

    def __init__(self):
        super().__init__()
        self.Resizable = True
        self.MinimumSize = ed.Size(650, 540)
        self.Height = 800
        self.Width = 900

        self.image_view = ef.ImageView()

        self.viewport_view = ef.ImageView()

        self.prompt_box = ef.TextArea()
        self.prompt_box.Height = 100
        self.prompt_box.Wrap = True
        self.prompt_box.Text = "a stunning view of a cluster of modular pavilions nestled within the lush Brazilian jungle the roof is built using woven bamboo elements surrounded by majestic mountains rising in the background and a serene river flowing in the foreground the trees are way taller than the pavilions earthy tones that blend harmoniously with the yellowish greens of the surrounding jungle volumetric sunlight goes across the jungle creating fascinating light rays 4k high resolution realistic render architectural visualization "

        self.negative_prompt_box = ef.TextArea()
        self.negative_prompt_box.Text = "low res, bad quality"
        self.negative_prompt_box.Height = 80
        self.negative_prompt_box.Wrap = True

        self.drop_down = ef.DropDown()
        self.drop_down.DataStore = [
            "stabilityai/sd-turbo",
            "SG161222/RealVisXL_V4.0",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ]
        self.drop_down.SelectedIndex = 0
        self.render_button = ef.Button()
        self.render_button.Text = "Render!"

        self.stepper = ef.NumericStepper()
        self.stepper.DecimalPlaces = 2
        self.stepper.MaxValue = 1
        self.stepper.MinValue = 0
        self.stepper.Value = 0.75
        # TODO : Strength changes
        
        self.seed_prompt_box = ef.NumericStepper()
        self.seed_prompt_box.DecimalPlaces = 0
        self.seed_prompt_box.MaxValue = 1_000_000
        self.seed_prompt_box.MinValue = 1
        self.seed_prompt_box.Value = randint(self.seed_prompt_box.MinValue, self.seed_prompt_box.MaxValue)

        self.inference_stepper = ef.NumericStepper()
        self.inference_stepper.DecimalPlaces = 0
        self.inference_stepper.MaxValue = 100
        self.inference_stepper.MinValue = 1
        self.inference_stepper.Value = 25
        
        self.DefaultButton = self.render_button
        self.AbortButton = ef.Button()

        self.save_button = ef.Button()
        self.save_button.Text = "Save Render"

        table_layout = ef.DynamicLayout()
        table_layout.MinimumSize = ed.Size(120, -1)
        table_layout.Spacing = ed.Size(6, 4)
        
        title = ef.Label()
        title.Text = "Properties"
        title.TextAlignment = ef.TextAlignment.Center

        ## --- Properties
        
        table_layout.Add(title, True, False)

        table_layout.BeginScrollable(ef.BorderType.NONE, None, None, True, False)
        table_layout.BeginVertical(ed.Padding(2), ed.Size(2, 4))

        controls = [
            ("Model",           self.drop_down,             "Some models are faster and some are better than others"),
            ("Strength",        self.stepper,               "The Strength of your AI"),
            ("Steps",           self.inference_stepper,     "Inference Steps are affect speed and quality"),
            ("Seed",            self.seed_prompt_box,       "Using the same seed makes the image generation similar"),
            ("Prompt",          self.prompt_box,            "The positive prompt is the goal of the AI"),
            ("Negative Prompt", self.negative_prompt_box,   "The negative prompt is ..."),

            (None,              None,                       None),

            (None,              self.render_button,         None),
            (None,              self.save_button,           None),
        ]

        for control_data in controls:
            if (control_data[0] != None):
                title = ef.Label()
                title.Height = 28
                title.VerticalAlignment = ef.VerticalAlignment.Bottom
                title.Wrap = ef.WrapMode.NONE
                title.HorizontalAlign = ef.HorizontalAlign.Left
                title.Text = control_data[0]
                table_layout.Add(title, True, False)

            if (control_data[2] != None):
                description = ef.Label()
                description.TextColor = ed.Colors.DimGray
                description.Wrap = ef.WrapMode.NONE
                description.HorizontalAlign = ef.HorizontalAlign.Left
                description.Text = control_data[2]
                table_layout.Add(description, True, False)

            table_layout.Add(control_data[1], True, False)

        table_layout.AddSpace(True, True)

        table_layout.EndScrollable()
        table_layout.EndVertical()
        
        self.tab_layout = ef.TabControl()
        self.tab_layout.Width = 520
        self.tab_layout.Height = 520
        self.render_page = ef.TabPage(self.image_view)
        self.render_page.Text = "AI Render"

        self.viewport_page = ef.TabPage(self.viewport_view)
        self.viewport_page.Text = "Viewport"
    
        self.tab_layout.Pages.Add(self.render_page)
        self.tab_layout.Pages.Add(self.viewport_page)

        ## --- Layout

        layout = ef.DynamicLayout()
        layout.Spacing = ed.Size(6, 6)
        layout.Padding = ed.Padding(6)
        layout.BeginHorizontal()
        layout.Add(self.tab_layout, True, True)
        layout.Add(table_layout, False, True)
        layout.EndHorizontal()

        self.Padding = ed.Padding(6)
        self.Content = layout
        self.Focus()

        def set_viewport(sender, args):
            bitmap = sc.doc.Views.ActiveView.CaptureToBitmap(Size(512,512), False, False, False)
            self.set_viewport_image(EtoExtensions.ToEto(bitmap))

        self.Shown += set_viewport

        def on_close(sender, args):
            self.dispose()

        self.Closed += on_close

    
    def dispose(self):
        try:
            if self.pipe:
                self.pipe.dispose()
        except:
            pass

    def get_seed(self):
        return self.seed_prompt_box.Value

    def get_steps(self):
        return self.inference_stepper.Value

    def get_prompt(self):
        return self.prompt_box.Text

    def get_negative_prompt(self):
        return self.negative_prompt_box.Text

    def set_render_image(self, bitmap:ed.Bitmap):
        self.image_view.Image = bitmap
        self.tab_layout.SelectedPage = self.render_page
        self.Invalidate(True)

    def set_viewport_image(self, bitmap:ed.Bitmap):
        self.viewport_view.Image = bitmap
        self.tab_layout.SelectedPage = self.viewport_page
        self.Invalidate(True)

    def get_model(self) -> str:
        return self.drop_down.SelectedValue


if __name__ == "__main__":
    dialog = RenderView()

    parent = RhinoEtoApp.MainWindowForDocument(sc.doc)
    EtoExtensions.ShowSemiModal(dialog, sc.doc, parent)