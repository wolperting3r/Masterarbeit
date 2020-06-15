def param_filename(parameters, include_plotdata=False, plotdata_as_data=False):
    # Generate filename string
    filename_string = ''
    # Namenskonvention
    old = False
    for key, value in parameters.items():
        if key == 'layers':
            filename_string = filename_string + '_' + '-'.join(str(e) for e in value)
        elif key == 'stencil_size':
            filename_string = filename_string + '_' + str(value[0]) + 'x' + str(value[1])
        elif key == 'equal_kappa':
            if old:
                filename_string = filename_string + '_' + ('eqk' if value else 'eqr')
            else:
                filename_string = filename_string
        elif key == 'negative':
            if old:
                filename_string = filename_string + '_' + ('neg' if value else 'pos')
            else:
                filename_string = filename_string
        elif key == 'angle':
            if old:
                filename_string = filename_string + '_' + ('ang' if value else 'nag')
            else:
                filename_string = filename_string
        elif key == 'rotate':
            filename_string = filename_string + '_' + ('rot' if value else 'nrt')
        elif key == 'smear':
            if old:
                filename_string = filename_string + '_' + ('smr' if value else 'nsm')
            else:
                filename_string = filename_string
        elif key == 'hf':
            # filename_string = filename_string + '_' + ('hef' if value else 'nhf')
            filename_string = filename_string
        elif key == 'addstring':
            filename_string = filename_string
        elif (key == 'plotdata') and (not include_plotdata):
            filename_string = filename_string
        elif key == 'hf_correction':
            if old:
                filename_string = filename_string + '_' + ('hfc' if value else 'nhc')
            else:
                filename_string = filename_string
        elif key == 'filename':
            filename_string = filename_string
        elif (key == 'data') and (plotdata_as_data):
            filename_string = filename_string + '_' + parameters['plotdata']
        elif key == 'epochs':
            filename_string = filename_string + '_' + str(value)
        elif key == 'network':
            filename_string = filename_string + '_' + str(value)
        elif key == 'flip':
            filename_string = filename_string + '_' + ('flp' if value else 'nfp')
        elif key == 'bias':
            filename_string = filename_string + '_' + ('bia' if value else 'nbi')
        elif key == 'cut':
            filename_string = filename_string + '_' + ('cut' if value else 'nct')
        elif ((key == 'dshift') and (parameters['dshift'] != '0')):
            filename_string = filename_string + '_' + 'dshift' + str(value)
        elif ((key == 'shift') and (parameters['shift'] != 0)):
            filename_string = filename_string + '_' + 'shift' + str(value)
        elif ((key == 'interpolate') and (parameters['interpolate'] != 0)):
            filename_string = filename_string + '_' + 'int' + str(value)
        else:
            if old:
                filename_string = filename_string + '_' + str(value)
            else:
                filename_string = filename_string
    # filename_string = filename_string + '_' + '_flat_e'
    return filename_string
