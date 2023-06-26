BOLD_RED = "\033[1;31m"
BOLD_GREEN = "\033[1;32m"
BOLD_YELLOW = "\033[1;33m"
BOLD_BLUE = "\033[1;34m"
END = "\033[0m"


def print_in_bold_red(words):
	print(BOLD_RED + words + END)

def print_in_bold_green(words):
	print(BOLD_GREEN + words + END)