

def parse_bool(s):
    # print("parse_bool: ", s)
    if s == "True":
        return True
    elif s == "False":
        return False
    elif s == "None":
        return None
    else:
        raise ValueError("Not a boolean value")
    

def parse_int_list(l):
    if l == "|":
        return []
    return [int(x) for x in l.split(",")]



def parse_str_list(l):
    if l == "|":
        return []
    return l.split(",")
