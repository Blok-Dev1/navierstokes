sph: sph.c
	gcc -fopenmp sph.c -lm -o sph `sdl-config --cflags --libs`

clean:
	rm sph
