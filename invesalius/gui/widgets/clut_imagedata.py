import glob
import math
import os

import wx

HISTOGRAM_LINE_COLOUR = (128, 128, 128)
HISTOGRAM_FILL_COLOUR = (64, 64, 64)
HISTOGRAM_LINE_WIDTH = 1

DEFAULT_COLOUR = (0, 0, 0)

TEXT_COLOUR = (255, 255, 255)
BACKGROUND_TEXT_COLOUR_RGBA = (255, 0, 0, 128)

GRADIENT_RGBA = 0.75 * 255

LINE_COLOUR = (128, 128, 128)
LINE_WIDTH = 2
RADIUS = 5

PADDING = 2


class CLUTEvent(wx.PyCommandEvent):
    def __init__(self, evtType, id, nodes):
        wx.PyCommandEvent.__init__(self, evtType, id)
        self.nodes = nodes

    def GetNodes(self):
        return self.nodes


# Occurs when CLUT point is changing
myEVT_CLUT_POINT_MOVE = wx.NewEventType()
EVT_CLUT_POINT_MOVE = wx.PyEventBinder(myEVT_CLUT_POINT_MOVE, 1)


class Node(object):
    def __init__(self, value, colour):
        self.value = value
        self.colour = colour

    def __cmp__(self, o):
        return cmp(self.value, o.value)

    def __repr__(self):
        return "(%d %s)" % (self.value, self.colour)


class CLUTImageDataWidget(wx.Panel):
    """
    Widget used to config the Lookup table from imagedata.
    """
    def __init__(self, parent, id, histogram, init, end, wl, ww):
        super(CLUTImageDataWidget, self).__init__(parent, id)

        self.SetMinSize((300, 200))

        self.histogram = histogram

        self._init = init
        self._end = end

        self.i_init = init
        self.i_end = end

        self._range = 0.05 * (end - init)

        self.wl = wl
        self.ww = ww

        wi = wl - ww / 2
        wf = wl + ww / 2

        self.nodes = [Node(wi, (0, 0, 0)),
                     Node(wf, (255, 255, 255))]

        self.middle_pressed = False
        self.right_pressed = False
        self.left_pressed = False

        self.selected_node = None
        self.last_selected = None

        self._d_hist = []

        self._build_drawn_hist()

        self.__bind_events_wx()

    def __bind_events_wx(self):
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackGround)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)

        self.Bind(wx.EVT_MOTION, self.OnMotion)

        self.Bind(wx.EVT_MOUSEWHEEL, self.OnWheel)
        self.Bind(wx.EVT_MIDDLE_DOWN, self.OnMiddleClick)
        self.Bind(wx.EVT_MIDDLE_UP, self.OnMiddleRelease)

        self.Bind(wx.EVT_LEFT_DOWN, self.OnClick)
        self.Bind(wx.EVT_LEFT_UP, self.OnRelease)
        self.Bind(wx.EVT_LEFT_DCLICK, self.OnDoubleClick)

        self.Bind(wx.EVT_RIGHT_DOWN, self.OnRightClick)

    def _build_drawn_hist(self):
        #w, h = self.GetVirtualSize()
        w = len(self.histogram)
        h = 1080

        x_init = self._init
        x_end = self._end

        y_init = 0
        y_end = math.log(self.histogram.max() + 1)

        prop_x = (w) * 1.0 / (x_end - x_init)
        prop_y = (h) * 1.0 / (y_end - y_init)

        self._d_hist = []
        for i in xrange(w):
            x = i / prop_x + x_init - 1
            if self.i_init <= x < self.i_end:
                try:
                    y = math.log(self.histogram[int(x - self.i_init)] + 1) * prop_y
                except IndexError:
                    print x, self.histogram.shape, x_init, self.i_init, self.i_end

                self._d_hist.append((i, y))

    def _interpolation(self, x):
        f = math.floor(x)
        c = math.ceil(x)
        h = self.histogram

        if f != c:
            return h[f] + (h[c] - h[f]) / (c - f) * (x - f)
        else:
            return h[int(x)]

    def OnEraseBackGround(self, evt):
        pass

    def OnSize(self, evt):
        #self._build_drawn_hist()
        self.Refresh()
        evt.Skip()

    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.SetBackground(wx.Brush('Black'))
        dc.Clear()

        self.draw_histogram(dc)
        self.draw_gradient(dc)

        if self.last_selected is not None:
            self.draw_text(dc)

    def OnWheel(self, evt):
        """
        Increase or decrease the range from hounsfield scale showed. It
        doesn't change values in preset, only to visualization.
        """
        direction = evt.GetWheelRotation() / evt.GetWheelDelta()
        init = self._init - direction * self._range
        end = self._end + direction * self._range
        self.SetRange(init, end)
        self.Refresh()

    def OnMiddleClick(self, evt):
        self.middle_pressed = True
        self.last_x = self.pixel_to_hounsfield(evt.GetX())

    def OnMiddleRelease(self, evt):
        self.middle_pressed = False

    def OnClick(self, evt):
        px, py = evt.GetPositionTuple()
        self.left_pressed = True
        self.selected_node = self.get_node_clicked(px, py)
        self.last_selected = self.selected_node
        if self.selected_node is not None:
            self.Refresh()

    def OnRelease(self, evt):
        self.left_pressed = False
        self.selected_node = None

    def OnDoubleClick(self, evt):
        w, h = self.GetVirtualSize()
        px, py = evt.GetPositionTuple()

        # Verifying if the user double-click in a node-colour.
        selected_node = self.get_node_clicked(px, py)
        if selected_node:
            # The user double-clicked a node colour. Give the user the
            # option to change the color from this node.
            colour_dialog = wx.GetColourFromUser(self, (0, 0, 0))
            if colour_dialog.IsOk():
                r, g, b = colour_dialog.Get()
                selected_node.colour = r, g, b
                self._generate_event()
        else:
            # The user doesn't clicked in a node colour. Creates a new node
            # colour with the DEFAULT_COLOUR
            vx = self.pixel_to_hounsfield(px)
            node = Node(vx, DEFAULT_COLOUR)
            self.nodes.append(node)
            self._generate_event()

        self.Refresh()

    def OnRightClick(self, evt):
        w, h = self.GetVirtualSize()
        px, py = evt.GetPositionTuple()
        selected_node = self.get_node_clicked(px, py)

        if selected_node:
            self.nodes.remove(selected_node)
            self._generate_event()
            self.Refresh()

    def OnMotion(self, evt):
        if self.middle_pressed:
            x = self.pixel_to_hounsfield(evt.GetX())
            dx = x - self.last_x
            init = self._init - dx
            end = self._end - dx
            self.SetRange(init, end)
            self.Refresh()
            self.last_x = x

        # The user is dragging a colour node
        elif self.left_pressed and self.selected_node:
            x = self.pixel_to_hounsfield(evt.GetX())
            print x, self.selected_node, type(x)
            self.selected_node.value = float(x)
            self.Refresh()

            # A point in the preset has been changed, raising a event
            self._generate_event()

    def draw_histogram(self, dc):
        w, h = self.GetVirtualSize()
        ctx = wx.GraphicsContext.Create(dc)

        ctx.SetPen(wx.Pen(HISTOGRAM_LINE_COLOUR, HISTOGRAM_LINE_WIDTH))
        ctx.SetBrush(wx.Brush(HISTOGRAM_FILL_COLOUR))

        path = ctx.CreatePath()
        xi, yi = self._d_hist[0]
        path.MoveToPoint(xi, h - yi)
        for x, y in self._d_hist:
            path.AddLineToPoint(x, h - y)

        ctx.Translate(self.hounsfield_to_pixel(self.i_init), 0)
        ctx.Translate(0, h)
        ctx.Scale(w * 1.0 / (self._end - self._init), h / 1080.)
        ctx.Translate(0, -h)
        #ctx.Translate(0, h * h/1080.0 )
        ctx.PushState()
        ctx.StrokePath(path)
        ctx.PopState()
        path.AddLineToPoint(x, h)
        path.AddLineToPoint(xi, h)
        path.AddLineToPoint(*self._d_hist[0])
        ctx.FillPath(path)

    def draw_gradient(self, dc):
        w, h = self.GetVirtualSize()
        ctx = wx.GraphicsContext.Create(dc)
        knodes = sorted(self.nodes)
        for ni, nj in zip(knodes[:-1], knodes[1:]):
            vi = round(self.hounsfield_to_pixel(ni.value))
            vj = round(self.hounsfield_to_pixel(nj.value))

            path = ctx.CreatePath()
            path.AddRectangle(vi, 0, vj - vi, h)

            ci = ni.colour + (GRADIENT_RGBA,)
            cj = nj.colour + (GRADIENT_RGBA,)
            b = ctx.CreateLinearGradientBrush(vi, h,
                                              vj, h,
                                              ci, cj)
            ctx.SetBrush(b)
            ctx.SetPen(wx.TRANSPARENT_PEN)
            ctx.FillPath(path)

            self._draw_circle(vi, ni.colour, ctx)
            self._draw_circle(vj, nj.colour, ctx)

    def _draw_circle(self, px, color, ctx):
        w, h = self.GetVirtualSize()

        path = ctx.CreatePath()
        path.AddCircle(px, h / 2, RADIUS)

        path.AddCircle(px, h / 2, RADIUS)
        ctx.SetPen(wx.Pen('white', LINE_WIDTH + 1))
        ctx.StrokePath(path)

        ctx.SetPen(wx.Pen(LINE_COLOUR, LINE_WIDTH - 1))
        ctx.SetBrush(wx.Brush(color))
        ctx.StrokePath(path)
        ctx.FillPath(path)

    def draw_text(self, dc):
        w, h = self.GetVirtualSize()
        ctx = wx.GraphicsContext.Create(dc)

        value = self.last_selected.value

        x = self.hounsfield_to_pixel(value)
        y = h / 2

        font = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
        font.SetWeight(wx.BOLD)
        font = ctx.CreateFont(font, TEXT_COLOUR)
        ctx.SetFont(font)

        text = 'Value: %-6d' % value

        wt, ht = ctx.GetTextExtent(text)

        wr, hr = wt + 2 * PADDING, ht + 2 * PADDING
        xr, yr = x + RADIUS, y - RADIUS - hr

        if xr + wr > w:
            xr = x - RADIUS - wr
        if yr < 0:
            yr = y + RADIUS

        xf, yf = xr + PADDING, yr + PADDING
        ctx.SetBrush(wx.Brush(BACKGROUND_TEXT_COLOUR_RGBA))
        ctx.SetPen(wx.Pen(BACKGROUND_TEXT_COLOUR_RGBA))
        ctx.DrawRectangle(xr, yr, wr, hr)
        ctx.DrawText(text, xf, yf)

    def _generate_event(self):
        evt = CLUTEvent(myEVT_CLUT_POINT_MOVE, self.GetId(), self.nodes)
        self.GetEventHandler().ProcessEvent(evt)

    def hounsfield_to_pixel(self, x):
        w, h = self.GetVirtualSize()
        p = (x - self._init) * w * 1.0 / (self._end - self._init)
        print "h->p", x, p, type(p)
        return p

    def pixel_to_hounsfield(self, x):
        w, h = self.GetVirtualSize()
        prop_x = (self._end - self._init) / (w * 1.0)
        p = x * prop_x + self._init
        print "p->h", x, p
        return p

    def get_node_clicked(self, px, py):
        w, h = self.GetVirtualSize()
        for n in self.nodes:
            x = self.hounsfield_to_pixel(n.value)
            y = h / 2

            if ((px - x)**2 + (py - y)**2)**0.5 <= RADIUS:
                return n

        return None

    def SetRange(self, init, end):
        """
        Sets the range from hounsfield
        """
        self._init, self._end = init, end
        print self._init, self._end
        #self._build_drawn_hist()
