def param_filename(parameters, include_plotdata=False, plotdata_as_data=False):
    # Generate filename string
    filename_string = ''
    for key, value in parameters.items():
        if key == 'layers':
            filename_string = filename_string + '_' + '-'.join(str(e) for e in value)
        elif key == 'stencil_size':
            filename_string = filename_string + '_' + str(value[0]) + 'x' + str(value[1])
        elif key == 'equal_kappa':
            filename_string = filename_string + '_' + ('eqk' if value else 'eqr')
        elif key == 'negative':
            filename_string = filename_string + '_' + ('neg' if value else 'pos')
        elif key == 'angle':
            filename_string = filename_string + '_' + ('ang' if value else 'nag')
        elif key == 'rotate':
            filename_string = filename_string + '_' + ('rot' if value else 'nrt')
        elif key == 'smear':
            filename_string = filename_string + '_' + ('smr' if value else 'nsm')
        elif key == 'hf':
            # filename_string = filename_string + '_' + ('hef' if value else 'nhf')
            filename_string = filename_string
        elif (key == 'plotdata') and (not include_plotdata):
            filename_string = filename_string
        elif key == 'hf_correction':
            filename_string = filename_string + '_' + ('hfc' if value else 'nhc')
        elif key == 'filename':
            filename_string = filename_string
        elif (key == 'data') and (plotdata_as_data):
            filename_string = filename_string + '_' + parameters['plotdata']
        else:
            filename_string = filename_string + '_' + str(value)
    # filename_string = filename_string + '_' + '_flat_e'
    return filename_string
