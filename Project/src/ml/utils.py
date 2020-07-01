def param_filename(parameters, include_plotdata=False, plotdata_as_data=False):
    # Generate filename string
    filename_string = ''
    # Namenskonvention
    old = False
    for key, value in parameters.items():
        # Network related keys
        if key == 'layers':
            filename_string = filename_string + '_' + '-'.join(str(e) for e in value)
        elif key == 'angle':
            if old:
                filename_string = filename_string + '_' + ('ang' if value else 'nag')
            else:
                filename_string = filename_string
        elif key == 'rotate':
            filename_string = filename_string + '_' + ('rot' if value else 'nrt')
        elif key == 'addstring':
            filename_string = filename_string
        elif key == 'hf_correction':
            if old:
                filename_string = filename_string + '_' + ('hfc' if value else 'nhc')
            else:
                filename_string = filename_string
        elif key == 'hf':
            # filename_string = filename_string + '_' + ('hef' if value else 'nhf')
            filename_string = filename_string
        elif (key == 'plotdata') and (not include_plotdata):
            filename_string = filename_string
        elif key == 'filename':
            filename_string = filename_string
        elif key == 'epochs':
            filename_string = filename_string + '_' + str(value)
        elif key == 'network':
            filename_string = filename_string + '_' + str(value)
        elif key == 'bias':
            filename_string = filename_string + '_' + ('bia' if value else 'nbi')
        elif key == 'cut':
            filename_string = filename_string + '_' + ('cut' if value else 'nct')
        elif key == 'custom_loss':
            filename_string = filename_string + '_' + ('cls' if value else 'ncl')
        elif ((key == 'shift') and (parameters['shift'] != 0)):
            filename_string = filename_string + '_' + 'shift' + str(value)
        elif key == 'edge':
            filename_string = filename_string + '_' + ('edg' if value else 'ned')
        elif (key == 'batch_size') and (parameters['batch_size'] != 128):
            filename_string = filename_string + '_' + 'bs' + str(value)
        elif key == 'flip':
            filename_string = filename_string + '_' + ('flp' if value else 'nfp')
        elif key == 'stencil_size':
            filename_string = filename_string + '_' + str(value[0]) + 'x' + str(value[1])
        else:
            if old:
                filename_string = filename_string + '_' + str(value)
            else:
                filename_string = filename_string


        if len(parameters['load_data']) > 0:
            if (key == 'load_data') and (len(parameters['load_data']) > 0):
                filename_string = filename_string + '_' + str(value)
        else:
            # Data related keys (not needed when data is explicitly set)
            if key == 'equal_kappa':
                if old:
                    filename_string = filename_string + '_' + ('eqk' if value else 'eqr')
                else:
                    filename_string = filename_string
            elif key == 'negative':
                if old:
                    filename_string = filename_string + '_' + ('neg' if value else 'pos')
                else:
                    filename_string = filename_string
            elif key == 'smear':
                if old:
                    filename_string = filename_string + '_' + ('smr' if value else 'nsm')
                else:
                    filename_string = filename_string
            elif (key == 'data') and (plotdata_as_data):
                filename_string = filename_string + '_' + parameters['plotdata']
            elif (key == 'data') and (not (plotdata_as_data)):
                filename_string = filename_string + '_' + parameters['data']
            elif key == 'gauss':
                filename_string = filename_string + ('_g' if value else '')
            elif ((key == 'dshift') and (parameters['dshift'] != '0')):
                filename_string = filename_string + '_' + 'dshift' + str(value)
            elif ((key == 'interpolate') and (parameters['interpolate'] != 0)):
                filename_string = filename_string + '_' + 'int' + str(value)
    # filename_string = filename_string + '_' + '_flat_e'
    return filename_string
