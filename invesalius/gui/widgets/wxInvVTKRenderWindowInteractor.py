from vtk.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor
import wx

class wxInvVTKRenderWindowInteractor(wxVTKRenderWindowInteractor):
    def __init__(self, parent, ID, *args, **kw):
        super().__init__(parent, ID, *args, **kw)

    def OnSize(self, evt):
        """Handles the wx.EVT_SIZE event for
        wxVTKRenderWindowInteractor.
        """
        # event processing should continue (we call this before the
        # Render(), in case it raises an exception)
        evt.Skip()

        try:
            width, height = evt.GetSize()
        except:
            width = evt.GetSize().width
            height = evt.GetSize().height
        width *= self.GetContentScaleFactor()
        height *= self.GetContentScaleFactor()
        self._Iren.SetSize(int(width), int(height))
        self._Iren.ConfigureEvent()

        # this will check for __handle
        self.Render()

    def OnPaint(self,event):
        """Handles the wx.EVT_PAINT event for
        wxVTKRenderWindowInteractor.
        """

        # wx should continue event processing after this handler.
        # We call this BEFORE Render(), so that if Render() raises
        # an exception, wx doesn't re-call OnPaint repeatedly.
        event.Skip()

        dc = wx.PaintDC(self)

        width, height = self.GetSize()
        width *= self.GetContentScaleFactor()
        height *= self.GetContentScaleFactor()

        # make sure the RenderWindow is sized correctly
        # make sure the RenderWindow is sized correctly
        self._Iren.GetRenderWindow().SetSize((int(width), int(height)))

        # Tell the RenderWindow to render inside the wx.Window.
        if not self._wxVTKRenderWindowInteractor__handle:

            # on relevant platforms, set the X11 Display ID
            d = self.GetDisplayId()
            if d and self._wxVTKRenderWindowInteractor__has_painted:
                self._Iren.GetRenderWindow().SetDisplayId(d)

            # store the handle
            self._wxVTKRenderWindowInteractor__handle = self.GetHandle()
            # and give it to VTK
            self._Iren.GetRenderWindow().SetWindowInfo(str(self._wxVTKRenderWindowInteractor__handle))

            # now that we've painted once, the Render() reparenting logic
            # is safe
            self._wxVTKRenderWindowInteractor__has_painted = True

        self.Render()
