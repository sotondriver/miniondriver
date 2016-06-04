# MongoDb dependencies
from pymongo import MongoClient

from time import sleep

import sys, getopt, linecache, csv, progressbar

# Agent Constants
BLANK = ''

# ConfigFile Constants
C_HOST = 1
C_HOST_ADDR = 0
C_HOST_PORT = 1
C_DB_NAME = 2
C_DOC_NAME = 3

# Error Constants
OPTION_ERROR = 0
PARAM_ERROR = 1
NO_ARG = 2
MAND_ARG_MISSING = 3
READING_CONF_ERROR = 4

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

	insertData(intputFile, configFile, userName, password)

# Prints errors 
def printError(errorType):
	if errorType == OPTION_ERROR:
		print 'Options Error: One or more options inserted do not exist'
	elif errorType == PARAM_ERROR:
		print 'Parameters Error: One or more parameters are missing'	
	elif errorType == NO_ARG:
		print 'No arguments shutting down! Bye Bye!'
	elif errorType == MAND_ARG_MISSING:
		print 'Mandatory arguments are missing!'
	elif errorType == READING_CONF_ERROR:
		print 'Error occurred while reading configuration file!'	

	sys.exit(2)

# Prints help for the commands available
def printHelp():
	for help in COMMAND_HELP:
		print help
	sys.exit(2)

# Main insert data process
def insertData(iFile, cFile, uName, password):
	if iFile == BLANK or cFile == BLANK:
		printError(MAND_ARG_MISSING)

	# Read configuration file
	dbHost = readConfig(cFile, C_HOST)
	dbName = readConfig(cFile, C_DB_NAME)
	tableName = readConfig(cFile, C_DOC_NAME)

	# Read csv and insert into MongoDB
	mongoConnection = MongoClient(dbHost.split(',')[C_HOST_ADDR], int(dbHost.split(',')[C_HOST_PORT]))
	mongoDatabase = mongoConnection[dbName]
	mongoCollection = mongoDatabase[tableName]

	# Csv file reading
	csvFile = open(iFile, 'r')
	fileSize = open(iFile, 'r')
	csvReader = csv.DictReader(csvFile)
	row_count = len(fileSize.readlines()) - 1

	bar = progressbar.ProgressBar(maxval = row_count, widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	iterator = 0

	bar.start()
	for entry in csvReader:
		bar.update(iterator + 1)
		mongoCollection.insert_one(entry)
		iterator += 1
		sleep(0.1)
	bar.finish()
	

# Read configFile and return information desired
def readConfig(cFile, infoReq):
	linecache.clearcache()
	return linecache.getline(cFile, infoReq).rstrip('\n')	
		
if __name__ == '__main__':
		main(sys.argv[1:])	