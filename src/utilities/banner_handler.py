from art import text2art

def banner_decorator(func):
    def wrapper(*args, **kwargs):
        ascii_art = text2art("Phishguard")
        print(ascii_art)
        print("\t\t\t\t\t by: UncoveredTensor\n")
        func(*args, **kwargs)
    return wrapper
