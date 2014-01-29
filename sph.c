#include <stdlib.h>
#include <stdio.h>
#include <SDL.h>
#include <math.h>
#include <time.h>

// SDL vars
#define WIDTH 640
#define HEIGHT 480
#define BPP 4
#define DEPTH 32
// END SDL vars

int n = 700;
float h = 0.015;
float dt = 0.002;
int plotEach=1;
float xmin=0;
float xmax=1;
float ymin=0;
float ymax=1;
float g = -2;
float k = 1000;
float my = 3.0e-8;
float f = 20.3;
int maxiter=2500;
float density0=0;
float *px;
float *py;
float *vx;
float *vy;
float *mass;
float *density;
float *pressure;
float *fx;
float *fy;
int i,j,iter;

struct Point {
  float x;
  float y;
};

float w(float x, float y){
  float r2 = x*x+y*y;
  float r = sqrtf(r2);
  if (0 <= r && r <= h)
    return 315/(64*M_PI*powf(h,4))*powf((h*h-r2),3);
  else
    return 0;
}

struct Point dw(float x, float y){
  float r2 = x*x+y*y;
  float r = sqrtf(r2);
  if (0 <= r && r <= h){
    struct Point p;
    p.x=-945/(32*M_PI*powf(h,4))*x*powf(-h*h+r2,2);
    p.y=-945/(32*M_PI*powf(h,4))*y*powf(-h*h+r2,2);
    return p;
  }
  else{
    struct Point p;
    p.x=0;
    p.y=0;
    return p;
  }
}

float ddw(float x, float y){
  float r2 = x*x+y*y;
  float r = sqrtf(r2);
  if (0 <= r && r <= h)
    return -945/(16*M_PI*powf(h,4))*(powf(h,4)-4*h*h*r+3*r*r);
  else
    return 0;
}

float dist(int i, int j){
  float dx = px[j]-px[i];
  float dy = py[j]-py[i];
  return sqrtf(dx*dx+dy*dy);
}

float min(float x, float y){
  if(x<y)
    return x;
  else
    return y;
}
float max(float x, float y){
  if(x>y)
    return x;
  else
    return y;
}
float signum(float x){
  if(x==0)
    return 0;
  else if(x<0)
    return -1;
  else
    return 1;
}

float randf(){
  float r = (float)rand()/(float)RAND_MAX;
  return r;
}

void set_pixel(SDL_Surface *screen, int x, int y, Uint32 color)
{
  if(x<1||y<1||WIDTH<x||HEIGHT<y)
    return;
  Uint32 *pixmem32;
  
  pixmem32 = (Uint32*) screen->pixels  + WIDTH*(HEIGHT-y) + x;
  *pixmem32 = color;
}

void draw_circle(SDL_Surface *surface, int n_cx, int n_cy, int radius, Uint8 r, Uint8 g, Uint8 b)
{
    // if the first pixel in the screen is represented by (0,0) (which is in sdl)
    // remember that the beginning of the circle is not in the middle of the pixel
    // but to the left-top from it:
 
    double error = (double)-radius;
    double x = (double)radius -0.5;
    double y = (double)0.5;
    double cx = n_cx - 0.5;
    double cy = n_cy - 0.5;
    Uint32 color = SDL_MapRGB( surface->format, r, g, b );
 
    while (x >= y)
    {
        set_pixel(surface, (int)(cx + x), (int)(cy + y), color);
        set_pixel(surface, (int)(cx + y), (int)(cy + x), color);
 
        if (x != 0)
        {
            set_pixel(surface, (int)(cx - x), (int)(cy + y), color);
            set_pixel(surface, (int)(cx + y), (int)(cy - x), color);
        }
 
        if (y != 0)
        {
            set_pixel(surface, (int)(cx + x), (int)(cy - y), color);
            set_pixel(surface, (int)(cx - y), (int)(cy + x), color);
        }
 
        if (x != 0 && y != 0)
        {
            set_pixel(surface, (int)(cx - x), (int)(cy - y), color);
            set_pixel(surface, (int)(cx - y), (int)(cy - x), color);
        }
 
        error += y;
        ++y;
        error += y;
 
        if (error >= 0)
        {
            --x;
            error -= x;
            error -= x;
        }
    }
}

void DrawScreen(SDL_Surface* screen){ 
    if(SDL_MUSTLOCK(screen)){
        if(SDL_LockSurface(screen) < 0) return;
    }
    for(i=0; i<n; i++){
      int x = px[i]*WIDTH;
      int y = py[i]*WIDTH;
      if (x<1 || WIDTH<x || y<1 || HEIGHT<y)
        continue;
      draw_circle(screen, x, y, h*WIDTH, 254, 254, 254);
    }

    if(SDL_MUSTLOCK(screen)) SDL_UnlockSurface(screen);
    SDL_Flip(screen); 
    SDL_FillRect(screen,NULL, 0x000000);
}

int main(){
  px = malloc(sizeof(float)*n);
  py = malloc(sizeof(float)*n);
  vx = malloc(sizeof(float)*n);
  vy = malloc(sizeof(float)*n);
  mass = malloc(sizeof(float)*n);
  density = malloc(sizeof(float)*n);
  pressure = malloc(sizeof(float)*n);
  fx = malloc(sizeof(float)*n);
  fy = malloc(sizeof(float)*n);


  // initialize SDL
  SDL_Surface *screen;
  SDL_Event event;
  if (SDL_Init(SDL_INIT_VIDEO) < 0 ) return 1;
  if (!(screen = SDL_SetVideoMode(WIDTH, HEIGHT, DEPTH, SDL_FULLSCREEN|SDL_HWSURFACE))){
    SDL_Quit();
    return 1;
  }

  // initialize
  srand(time(NULL));
  for(i=0;i<n;i++){
    px[i]=0.4*(i % (int)sqrt(n) / sqrt(n))+0.3+randf()/100;
    py[i]=0.4*(float)i/(float)n+0.05+randf()/100;
    vx[i]=0;
    vy[i]=0;
    mass[i]=0.2;
    density[i]=0;
    pressure[i]=0;
  }

  for(iter=0;iter<maxiter;iter++){

    // compute density
    #pragma omp parallel for private(j) schedule(static) num_threads(4)
    for(i=0; i<n; i++){
      density[i] = 0;
      for(j=0; j<n; j++){
        if(dist(i,j)<=h){
          density[i] += mass[j]*w(px[i]-px[j], py[i]-py[j]);
        }
      }
      pressure[i] = k*(density[i]-density0);
    }

    // compute interactions
    #pragma omp parallel for private(j) schedule(static) num_threads(4)
    for(i=0;i<n;i++){

      float fpx = 0;
      float fpy = 0;
      float fvx = 0;
      float fvy = 0;
      for(j=0;j<n;j++){
        // check for kernel proximity
        if(dist(i,j)<=h){
          struct Point p = dw(px[i]-px[j],py[i]-py[j]);
          fpx -= mass[j]*(pressure[i]+pressure[j])/(2*density[j])*p.x;
          fpy -= mass[j]*(pressure[i]+pressure[j])/(2*density[j])*p.y;

          fvx -= my*mass[j]*(vx[j]-vx[i])/density[j]*ddw(px[i]-px[j],py[i]-py[j]);
          fvy -= my*mass[j]*(vy[j]-vy[i])/density[j]*ddw(px[i]-px[j],py[i]-py[j]);
        }
      }

      fx[i]=fpx+fvx;
      fy[i]=fpy+fvy+g;

      if(px[i] < xmin){ // outside WEST
        fy[i] -= signum(fy[i])*min(f*abs(fx[i]),abs(fy[i]));
        fx[i] = (xmin-px[i]-vx[i]*dt)/(dt*dt);
      }
      else if(xmax < px[i]){ // outside EAST
        fy[i] -= signum(fy[i])*min(f*abs(fx[i]),abs(fy[i]));
        fx[i] = (xmax-px[i]-vx[i]*dt)/(dt*dt);
      }
      else if(py[i] < ymin){ // outside SOUTH
        fx[i] -= signum(fx[i])*min(f*abs(fy[i]),abs(fx[i]));
        fy[i] = (ymin-py[i]-vy[i]*dt)/(dt*dt);
      }
      else if(xmax < px[i]){ // outside NORTH
        fx[i] -= signum(fx[i])*min(f*abs(fy[i]),abs(fx[i]));
        fy[i] = (ymax-py[i]-vy[i]*dt)/(dt*dt);
      }

      // do euler steps
      vx[i] += fx[i]*dt;
      vy[i] += fy[i]*dt;

      px[i] += vx[i]*dt;
      py[i] += vy[i]*dt;
    }
    DrawScreen(screen);
    while(SDL_PollEvent(&event)) 
    {      
      switch (event.type) 
      {
        case SDL_QUIT:
          SDL_Quit();
          return 0;
	        break;
        case SDL_KEYDOWN:
          SDL_Quit();
          return 0;
          break;
      }
    }
  }

  SDL_Quit();
  return 0;
}
