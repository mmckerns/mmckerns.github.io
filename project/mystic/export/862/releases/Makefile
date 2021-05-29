# CURRENT RELEASE:
#---------------------------------------------------
NPATCH_0_2a1 = 2
#---------------------------------------------------

all: patch release zip exe clean
tgz: patch release clean
src: patch release zip clean
win: patch zip exe clean
exe: installer distclean

patch:
	patch -p0 -i mystic-0.2a1-patch${NPATCH_0_2a1}.diff

zip:
	zip -r mystic-0.2a1-p${NPATCH_0_2a1}.zip mystic-0.2a1

release:
	tar -cvzf mystic-0.2a1-p${NPATCH_0_2a1}.tgz mystic-0.2a1

installer:
	cd mystic-0.2a1; python setup.py bdist --format=wininst; cd ..
	cd mystic-0.2a1; cd dist; mv -f mystic-0.2a1*exe ..; cd ..; cd ..
	cd mystic-0.2a1; mv -f mystic-0.2a1*exe ..; cd ..
	mv -f mystic-0.2a1*exe mystic-0.2a1-p${NPATCH_0_2a1}.win32.exe

distclean:
	cd mystic-0.2a1; rm -rf build dist mystic.egg-info; cd ..

clean:
	cd mystic-0.2a1; cd examples; rm -f test_mogi*; cd ..; cd ..
	cd mystic-0.2a1; cd mystic; rm -f tools*; cd ..; cd ..
	cd mystic-0.2a1; cd mystic; rm -f differential_evolution*; cd ..; cd ..
	svn up


# PRIOR RELEASES:
#---------------------------------------------------
NPATCH_0_1a2 = 0
#---------------------------------------------------

release_0_1a2:
	tar -cvzf mystic-0.1a2.tgz mystic-0.1a2

clean_0_1a2:
	svn up


#---------------------------------------------------
NPATCH_0_1a1 = 3
#---------------------------------------------------
patch_0_1a1:
	patch -p0 -i mystic-0.1a1-patch${NPATCH_0_1a1}.diff

release_0_1a1:
	tar -cvzf mystic-0.1a1-p${NPATCH_0_1a1}.tgz mystic-0.1a1

clean_0_1a1:
	cd mystic-0.1a1; cd examples; rm -f README example*; cd ..; cd ..
	cd mystic-0.1a1; cd mystic; rm -f scipy_optimize*; cd ..; cd ..
	cd mystic-0.1a1; rm -f setup.py; cd ..
	svn up

