'''Quantopian is a platform that allows us to write and back-test Python-powered trading strategies very easily

        -It adds a GUI layer on top of the Zipline back-testing library
        -Quantopian provides capital allocations to some users who meet the requirements
        -powered by Zipline, Alphalens, and Pyfolio

        -

'''






def initialize(context):
    context.aapl = sid(24)

def handle_data(context, data):