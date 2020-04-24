from views import app

if __name__ == '__main__' :
    ## True: when developping
    ## False: when app lauchned
	app.run(debug=True, host='0.0.0.0', port=8090)
