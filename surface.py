""" Module for processing surfaces from Olympus LEXT software

Author: Rich Winslow
Principal Investigators: Prof. Paul Wright, Prof. James Evans
University: University of California, Berkeley

classes:
    Surface:
        methods:
            __init__(*filename)
            parse_waviness()
            parse_roughness()
            calculate_metrics()
            plot_primary()
            plot_metrics(*metric)
            plot_section(*index)

"""

import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Surface:

    """ Generates surface profile components and metrics

    Produces primary, waviness, and roughness profiles from a CSV containing
    height information from Olympus LEXT software

    Note that the LEXT software starts indexing at 1 for row numbers, while
    Python starts indexing at 0 for when you're comparing sections of
    profiles

    glossary:
        self.npr = number of elements/points per row of data

    """

    def __init__(self, filepath, cutoff=80, sample_width=643):
        """ Opens file and processes all data

        Parses row by comma because input must be a CSV.

        All data from a LEXT-generated CSV is on a single line, so it is
        reshaped to a 2-D matrix from a 1-D vector.

        arguments:
            filepath = Path to data
            cutoff = Cutoff wavelength for low pass FFT filter
                default: 80 um
            sample_width = Width of sample area in microns (um)
                default: 643 um at 10x magnification on Olympus

        """

        self.filepath = filepath
        self.cutoff = cutoff
        self.sample_width = sample_width

        with open(filepath) as f:
            row = f.readlines()
            self.primary = [float(x) for x in row[0].split(',')
                            if x is not None and len(x) > 0]

            self.npr = int(numpy.sqrt(len(self.primary)))
            self.primary = numpy.reshape(self.primary, (self.npr, self.npr))

            self.parse_waviness()
            self.parse_roughness()
            self.calculate_metrics()

    def parse_waviness(self):
        """ Parse waviness from each row of the primary profile

        Computes the FFT of the primary profile line-by-line.

        To prevent non-zero values at the boundaries, the primary profile is
        extended at the beginning and end by a flipped version of itself.

        The dataset is all real valued, so the FFT is symmetric. Thus, the
        signal strength must be doubled to fit the data correctly.

        For waviness, a low-pass filter is used (allows low frequencies/long
        wavelength signals) to allow the wavelengths longer than the cutoff
        wavelength to contribute to the final waviness profile. All values
        outside the range of allowed values are set to zero.

        """

        self.waviness = []

        for i in range(self.npr):
            row = self.primary[i]
            profile = []
            flipped = row[::-1]

            profile.extend(flipped)
            profile.extend(row)
            profile.extend(flipped)

            f = numpy.array(numpy.fft.fft(profile))
            f[1:-1] = f[1:-1]*2

            self.wavelengths = []
            for j in range(1, self.npr):
                wavelength = 2*(3*self.sample_width)/j
                self.wavelengths.extend([wavelength])

                if (wavelength <= self.cutoff):
                    stop_index = j
                    break

            filtered = f
            filtered[stop_index:-1] = 0

            self.waviness.append(numpy.real(numpy.fft.ifft(filtered))
                                 [self.npr:2*self.npr].tolist())

    def parse_roughness(self):
        """ Parse roughness from  primary and waviness profiles

        Runs through each row in primary and waviness profiles and finds the
        difference between them to get the roughness

        """

        self.roughness = []
        for i in range(self.npr):
            self.roughness.append(self.primary[i] - self.waviness[i])

    def calculate_metrics(self):
        """ Calculate metrics for each row of waviness and roughness

        Calculates:
            Wa = Average waviness
            Ra = Average roughness

        """

        Wa = [sum(numpy.abs(self.waviness[i]))/self.npr
              for i in range(self.npr)]
        Ra = [sum(numpy.abs(self.roughness[i]))/self.npr
              for i in range(self.npr)]

        self.metrics = {'Wa': Wa, 'Ra': Ra}

    def plot_primary(self):
        """ Plots top down view of primary surface """

        im = plt.imshow(self.primary, extent=[0, 643, 0, 643], cmap=cm.jet)
        plt.colorbar(im, label='Height (um)')

        plt.xlabel('X Position (um)')
        plt.ylabel('Y Position (um)')

        plt.show()

    def plot_section(self, index):
        """ Plots cross section of profile with waviness and roughness on plot

        """

        X = numpy.linspace(0, self.sample_width, self.npr)

        plt.plot(X, self.primary[index], label='Primary')
        plt.plot(X, self.waviness[index], label='Waviness')
        plt.plot(X, self.roughness[index], label='Roughness')

        plt.xlim(0, self.sample_width)
        plt.xlabel('Position (um)')
        plt.ylabel('Height (um)')
        plt.legend(ncol=3, loc='upper center')
        plt.show()

    def plot_metrics(self, metric):
        """ Plots data from one of the metrics and center around zero """

        centered = (self.metrics[metric] -
                    (sum(self.metrics[metric])/float(len(
                     self.metrics[metric]))))

        X = numpy.linspace(0, self.sample_width, self.npr)
        plt.plot(X, centered)

        plt.gcf().subplots_adjust(bottom=0.3)
        plt.xlim(0, self.sample_width)
        plt.xlabel('Position (um)')
        plt.ylabel('Height (um)')
        plt.title(metric)
        plt.show()
