# MongoDb dependencies
from pymongo import MongoClient

import sys, getopt


# Error Constants
OPTION_ERROR = 0
PARAM_ERROR = 1
NO_ARG = 2
MAND_ARG_MISSING = 3

# Instruction Constant
COMMAND_HELP = ['-i\tInput file name from where wanted to read the data (CSV)', '-c\tConfiguration file where all the information about the database and structure is contained', '-u\t Username of database', '-p\tPassword of the database']

# Main user interface
def main(argv):
	if len(argv) == 0:
		printError(NO_ARG)

	try:
		opts, args = getopt.getopt(argv, 'h:i:c:u:p', ['help', 'inputFile=', 'configFile=', 'username=', 'password='])
	except getopt.GetoptError:
		printError(OPTION_ERROR)

	intputFile = ''
	configFile = ''
	userName = ''
	password = ''

	for opt, arg in opts:
		 if opt in ('-h', '--help'):
		 	printHelp()
		 elif opt in ('-i', '--input'):
		 	intputFile = arg
		 elif opt in ('-c', '--config'):
		 	configFile = arg
		 elif opt in ('-u', '--username'):
		 	userName = arg
		 elif opt in ('-p', '--password'):
		 	password = arg		
		 else:
		 	printError(PARAM_ERROR)

	insertData(intputFile, configFile)

# Prints errors 
def printError(errorType):
	if errorType == OPTION_ERROR:
		print 'Options Error: One or more options inserted do not exist'
	elif errorType == PARAM_ERROR:
		print 'Parameters Error: One or more parameters are missing'	
	elif errorType == NO_ARG:
		print 'No arguments shutting down! Bye Bye!'
	elif errorType ==MAND_ARG_MISSING:
		print 'Mandatory arguments are missing!'
		
	sys.exit(2)

# Prints help for the commands available
def printHelp():
	for help in COMMAND_HELP:
		print help

# Main insert data process
def insertData(iFile, cFile):
	print 'testing'

if __name__ == '__main__':
		main(sys.argv[1:])	