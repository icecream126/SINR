def parse_t_f(arg):
    """Used to create flags like --flag=True with argparse"""
    ua = str(arg).upper()
    if "TRUE".startswith(ua):
        return True
    elif "FALSE".startswith(ua):
        return False
    else:
        raise ValueError("Arg must be True or False")
