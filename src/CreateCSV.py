import sys

# Checks if this is __main__ so that sphinx doesn't run the code
# when generating docs.
if __name__ == "__main__":

    # Looks for the command flags.
    for arg in sys.argv:

        if (arg == "help"):
            
            file = open("HelpMessage.txt", 'r')
            helpLines = file.readlines()
            file.close()

            print("")
            for line in helpLines:
                print(line, end="")
            print("\n")