def param_filename(parameters):
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
        else:
            filename_string = filename_string + '_' + str(value)
    return filename_string
