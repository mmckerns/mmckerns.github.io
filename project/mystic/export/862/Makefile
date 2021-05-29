#---------------------------------------------------
VERSION	= 0.2a2.dev
#---------------------------------------------------

all: doc reminder

doc:
	cp -rf mystic/_math mystic/math
	cp -f branches/dummy/__init__.py mystic/scripts
	epydoc --config mystic.epydoc
	rm -rf mystic/math
	rm -f mystic/scripts/__init__.py
	mkdir html
	mv -f shared,mpi html/mystic-${VERSION}

# remind to search for /'</ and delete '
reminder:
	echo "........................................................................."
	echo "before posting: edit module-tree.html & models-module.html to remove (')s"
	echo "before posting: check that the version correct in all source locations"
	echo "........................................................................."

clean:
	cd mystic; mm; cd ..

restore: clean
	rm -rf html
