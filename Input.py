import os
import optparse

def get_cmdline_arguments():
    """
        Returns dictionary of command line arguments supplied to PhonoDark.
    """

    parser = optparse.OptionParser()
    parser.add_option('-x', action="store", default="",
            help="Initial xi value for which to run ConstH.")
    parser.add_option('-b', action="store", default="",
            help="Value of beta/Mpl")
    parser.add_option('-d', action="store", default="None",
            help="Order of magnitude for deviations from xi")
    parser.add_option('-s', action="store", default="+",
            help="sign of deviations from xi")  

    options_in, args = parser.parse_args()

    options = vars(options_in)

    cmd_input_okay = False
    if options['x'] != '' and options['b'] != '':
        cmd_input_okay = True

    return options, cmd_input_okay
