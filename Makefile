sph: sph.c
	gcc -fopenmp -finline-functions -O2 sph.c -lm -o sph `sdl-config --cflags --libs`

clean:
	rm sph
