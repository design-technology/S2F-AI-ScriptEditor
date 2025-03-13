import Rhino
import scriptcontext as sc
# subscribing to any event within the rhino viewport
# The CalculateBoundingBox event is part of Rhino's display pipeline and fires whenever Rhino needs to recalculate what's visible in the viewport
class ViewportMonitor:
    def __init__(self):
        # Register event handlers
        Rhino.Display.DisplayPipeline.CalculateBoundingBox += self.on_viewport_changed
        self.Enabled = True
        print("Monitoring started")
    
    def Disable(self):
        # Unregister event handlers
        Rhino.Display.DisplayPipeline.CalculateBoundingBox -= self.on_viewport_changed
        self.Enabled = False
    
    def on_viewport_changed(self, sender, e):
        print("Viewport changed")

def toggle_monitoring():
    # Check if already monitoring
    # sc.sticky (a persistent dictionary in Rhino) to keep the monitor alive between script runs
    if sc.sticky.has_key('viewport_monitor'):
        # Stop monitoring
        monitor = sc.sticky.pop('viewport_monitor')
        monitor.Disable()
        print("Monitoring stopped")
    else:
        # Start monitoring
        monitor = ViewportMonitor()
        sc.sticky['viewport_monitor'] = monitor

# Run when script is executed
if __name__ == '__main__':
    toggle_monitoring()