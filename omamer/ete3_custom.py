from __future__ import absolute_import
from __future__ import print_function
import re
import math

try:
    from numpy import isfinite as _isfinite, ceil
except ImportError:
    pass
else:
    isfinite = lambda n: n and _isfinite(n)

try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

import six
from six.moves import map
from six.moves import range
from six.moves import zip

from .qt import (QGraphicsRectItem, QGraphicsLineItem,
                 QGraphicsPolygonItem, QGraphicsEllipseItem,
                 QPen, QColor, QBrush, QPolygonF, QFont,
                 QPixmap, QFontMetrics, QPainter,
                 QRadialGradient, QGraphicsSimpleTextItem, QGraphicsTextItem,
                 QGraphicsItem, Qt,  QPointF, QRect, QRectF, QGraphicsSvgItem)

from .main import add_face_to_node, _Background, _Border, COLOR_SCHEMES


from ete3 import *


class _BarChartItem(QGraphicsRectItem):
    def __init__(self, values, deviations, width, height, colors, labels,
                 min_value, max_value, label_fsize, scale_fsize):
        QGraphicsRectItem.__init__(self, 0, 0, width, height)
        self.values = values
        self.colors = colors
        self.width = float(width)
        self.height = float(height)
        self.draw_border = True
        self.draw_grid = False
        self.draw_scale = True
        self.labels = labels
        self.max_value = max_value
        self.min_value = min_value
        self.deviations = deviations
        self.label_fsize = label_fsize
        self.scale_fsize = scale_fsize

        self.set_real_size()

    def set_real_size(self):
        label_height = 0
        scale_width = 0
        margin = 2

        if self.max_value is None:
            max_value = max([v+d for v,d in zip(self.values, self.deviations) if isfinite(v)])
        else:
            max_value = self.max_value

        if self.min_value is None:
            min_value = min([v+d for v,d in zip(self.values, self.deviations) if isfinite(v)])
        else:
            min_value = self.min_value

        if self.draw_scale:
            max_string = "% 7.2f" %max_value
            min_string = "% 7.2f" %min_value
            fm = QFontMetrics(QFont("Verdana", self.scale_fsize))
            max_string_metrics = fm.boundingRect(QRect(), \
                                                 Qt.AlignLeft, \
                                                 max_string)
            min_string_metrics = fm.boundingRect(QRect(), \
                                                 Qt.AlignLeft, \
                                                 min_string)
            scale_width = margin + max(max_string_metrics.width(),
                                             min_string_metrics.width())

        if self.labels:
            fm = QFontMetrics(QFont("Verdana", self.label_fsize))
            longest_label = sorted(self.labels, key=lambda x: len(x))[-1]
            label_height = fm.boundingRect(QRect(), Qt.AlignLeft, longest_label).width() + margin
            label_width = fm.height() * len(self.labels)
            self.width = max(label_width, self.width)

        self.setRect(0, 0, self.width + scale_width, self.height + label_height)


    def paint(self, p, option, widget):
        colors = self.colors
        values = self.values
        deviations = self.deviations
        p.setBrush(Qt.NoBrush)
        margin = 2
        spacer = 3
        spacing_length = (spacer*(len(values)-1))
        height = self.height

        if self.max_value is None:
            max_value = max([v+d for v,d in zip(values, deviations) if isfinite(v)])
        else:
            max_value = self.max_value

        if self.min_value is None:
            min_value = min([v+d for v,d in zip(values, deviations) if isfinite(v)])
        else:
            min_value = self.min_value

        plot_width = self.width
        plot_height = self.height

        x_alpha = float((plot_width - spacing_length) / (len(values)))
        if x_alpha < 1:
            raise ValueError("BarChartFace is too small")

        y_alpha = float ( (plot_height-3) / float(max_value - min_value) )
        x = 0
        y = 0

        # Mean and quartiles y positions
        mean_line_y = y + (plot_height / 2.0)
        line2_y = mean_line_y + (plot_height/4.0)
        line3_y = mean_line_y - (plot_height/4.0)

        if self.draw_border:
            p.setPen(QColor("black"))
            p.drawRect(x, y + 1, plot_width, plot_height)

        if self.draw_scale:
            p.setFont(QFont("Verdana", self.scale_fsize))
            font_height = QFontMetrics(p.font()).height()
            max_string = "% 7.2f" %max_value
            min_string = "% 7.2f" %min_value
            p.drawText(plot_width + margin, font_height-2, max_string)
            p.drawText(plot_width + margin, plot_height - 2, min_string)
            p.drawLine(plot_width + margin - 1, 1, plot_width + margin - 1, plot_height+1)
            p.drawLine(plot_width + margin - 1, 1, plot_width + margin + 2, 1)
            p.drawLine(plot_width + margin - 1, plot_height+1, plot_width + margin + 2, plot_height+1)

        if self.draw_grid:
            dashedPen = QPen(QBrush(QColor("#ddd")), 0)
            dashedPen.setStyle(Qt.DashLine)
            p.setPen(dashedPen)
            p.drawLine(x+1, mean_line_y, plot_width - 2, mean_line_y)
            p.drawLine(x+1, line2_y, plot_width - 2, line2_y )
            p.drawLine(x+1, line3_y, plot_width - 2, line3_y )

        # Draw bars
        p.setFont(QFont("Verdana", self.label_fsize))
        label_height = self.rect().height() - self.height
        label_width = QFontMetrics(p.font()).height()
        for pos in range(len(values)):
            # first and second X pixel positions
            x1 = x
            x = x1 + x_alpha + spacer

            std =  deviations[pos]
            val = values[pos]

            if self.labels:
                p.save()
                p.translate(x1, plot_height+2)
                p.rotate(90)
                p.drawText(0, -x_alpha, label_height, x_alpha, Qt.AlignVCenter, str(self.labels[pos]))
                #p.drawRect(0, -x_alpha, label_height, x_alpha)
                p.restore()

            # If nan value, skip
            if not isfinite(val):
                continue

            color = QColor(colors[pos])
            # mean bar high
            mean_y1     = int((val - min_value) * y_alpha)
            # Draw bar border
            p.setPen(QColor("black"))

            # Fill bar with custom color
            p.fillRect(x1, height - mean_y1, x_alpha, mean_y1, QBrush(color))

            # Draw error bars
            if std != 0:
                dev_up_y1   = int((val + std - min_value) * y_alpha)
                dev_down_y1 = int((val - std - min_value) * y_alpha)
                center_x = x1 + (x_alpha / 2)
                p.drawLine(center_x, plot_height - dev_up_y1, center_x, plot_height - dev_down_y1)
                p.drawLine(center_x + 1, plot_height - dev_up_y1, center_x -1, plot_height - dev_up_y1)
                p.drawLine(center_x + 1, plot_height - dev_down_y1, center_x -1, plot_height - dev_down_y1)