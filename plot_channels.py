from collections import defaultdict

from pylearn2.train_extensions import TrainExtension
import matplotlib.pyplot as plt

class PlotChannels(TrainExtension):

    def __init__(self, channels, legend_loc='upper right'):
        self.channels = channels
        self.legend_loc = legend_loc
        self.colors = 'bgrcmyk'
        plt.ion()

    def _channel_value(self, model, channel):
        c = model.monitor.channels[channel]
        return c.val_shared.get_value().item()

    def setup(self, model, dataset, algorithm):
        self.X = []
        self.Y = defaultdict(list)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.replot()
        self.ax.legend(loc=self.legend_loc)
        self.fig.canvas.draw()

    def replot(self):
        for i, c in enumerate(self.channels):
            self.ax.plot(self.Y[c], label=c, color=self.colors[i])

    def update(self, model):
        self.X.append(len(self.X)+1)
        for i, c in enumerate(self.channels):
            value = self._channel_value(model, c)
            self.Y[c].append(value)
        self.replot()
            
    def on_monitor(self, model, dataset, algorithm):
        self.update(model)
        self.fig.canvas.draw()
        self.fig.canvas.draw()