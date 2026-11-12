def relative_l1(reference, current):
    reference, current = reference.float(), current.float()
    return ((current - reference).abs().mean() / (current.abs().mean() + 1e-8)).item()


def fit_channels(previous, current):
    dims = [dim for dim in range(current.ndim) if dim != 1]
    previous, current = previous.float(), current.float()
    previous_mean, current_mean = previous.mean(dims, keepdim=True), current.mean(dims, keepdim=True)
    centered = previous - previous_mean
    slope = (centered * (current - current_mean)).mean(dims) / centered.square().mean(dims).clamp_min(1e-8)
    intercept = current_mean.flatten() - slope * previous_mean.flatten()
    return slope.tolist(), intercept.tolist()


def column_mean(rows):
    return [sum(column) / len(column) for column in zip(*rows, strict=True)]
