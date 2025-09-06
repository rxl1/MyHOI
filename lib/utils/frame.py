




def align_frame(total_dict):
    max_nframes = 0
    value = next(iter(total_dict.values())) # ndata, frames, _
    for _value in value:
        nframes = len(_value)
        if nframes > max_nframes:
            max_nframes = nframes
    ndata = len(value)