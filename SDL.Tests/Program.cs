using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static SDL2.SDL;

namespace SDL.Tests
{
    internal class Program
    {
        // SDL vars
        static int WIDTH = 640;
        static int HEIGHT = 480;
        static int BPP = 4;
        static int DEPTH = 32;
        // END SDL vars

        static int n = 800;
        static float h = 0.015f;
        static float dt = 0.002f;
        static int plotEach = 1;
        static float xmin = 0;
        static float xmax = 1;
        static float ymin = 0;
        static float ymax = 1;
        static float g = -2;
        static float k = 750;
        static float my = 3.0e-8f;
        static float f = 20.3f;
        static int maxiter = 2500;
        static float density0 = 0;
        static float[] px = new float[n];
        static float[] py = new float[n];
        static float[] vx = new float[n];
        static float[] vy = new float[n];
        static float[] mass = new float[n];
        static float[] density = new float[n];
        static float[] pressure = new float[n];
        static float[] fx = new float[n];
        static float[] fy = new float[n];
        static int i, j, iter, frame = 0;

        static Random rand = new Random(Environment.TickCount);
        struct Point
        {
            public float x;
            public float y;
        };

        static float w(float x, float y)
        {
            float r2 = x * x + y * y;
            double r = Math.Sqrt(r2);
            if (0 <= r && r <= h)
                return (float)(315 / (64 * Math.PI * Math.Pow(h, 4)) * Math.Pow((h * h - r2), 3));
            else
                return 0;
        }
        static Point dw(float x, float y)
        {
            float r2 = x * x + y * y;
            var r = Math.Sqrt(r2);

            if (0 <= r && r <= h)
            {
                Point p = new Point();
                p.x = (float)(-945 / (32 * Math.PI * Math.Pow(h, 4)) * x * Math.Pow(-h * h + r2, 2));
                p.y = (float)(-945 / (32 * Math.PI * Math.Pow(h, 4)) * y * Math.Pow(-h * h + r2, 2));
                return p;
            }
            else {
                Point p = new Point();
                p.x = 0;
                p.y = 0;
                return p;
            }
        }
        static float ddw(float x, float y)
        {
            float r2 = x * x + y * y;
            var r = Math.Sqrt(r2);
            if (0 <= r && r <= h)
                return (float)(-945 / (16 * Math.PI * Math.Pow(h, 4)) * (Math.Pow(h, 4) - 4 * h * h * r + 3 * r * r));
            else
                return 0;
        }

        static float min(float x, float y)
        {
            if (x < y)
                return x;
            else
                return y;
        }
        static float max(float x, float y)
        {
            if (x > y)
                return x;
            else
                return y;
        }
        static float signum(float x)
        {
            if (x == 0)
                return 0;
            else if (x < 0)
                return -1;
            else
                return 1;
        }
        
        static float randf()
        {
            //float r = (float)rand() / (float)RAND_MAX;
            var r = (float)rand.NextDouble();
            return r;
        }

        static float dist(int i, int j)
        {
            var dx = px[j] - px[i];
            var dy = py[j] - py[i];
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        static IntPtr window;
        static IntPtr renderer;

        static bool running = true;
        static void Main(string[] args)
        {
            Setup();

            InitSPH();

            for (iter = 0; iter < maxiter; iter++)
            {
                while (running)
                {
                    PollEvents();

                    RenderSPH();
                }
            }

            CleanUp();
        }

        private static void InitSPH()
        {
            // initialize
            for (i = 0; i < n; i++)
            {
                px[i] = (float)(0.4 * (i % (int)Math.Sqrt(n) / Math.Sqrt(n)) + 0.3 + randf() / 100);
                py[i] = (float)(0.4 * (float)i / (float)n + 0.05 + randf() / 100);
                vx[i] = 0;
                vy[i] = 0;
                mass[i] = 0.2f;
                density[i] = 0;
                pressure[i] = 0;
            }
        }

        private static void RenderSPH()
        {
            // compute density
            //#pragma omp parallel for private(j) schedule(static) num_threads(4)
            for (i = 0; i < n; i++)
            {
                density[i] = 0;
                for (j = 0; j < n; j++)
                {
                    if (dist(i, j) <= h)
                    {
                        density[i] += mass[j] * w(px[i] - px[j], py[i] - py[j]);
                    }
                }
                pressure[i] = k * (density[i] - density0);
            }

            // compute interactions
            //#pragma omp parallel for private(j) schedule(static) num_threads(4)
            for (i = 0; i < n; i++)
            {
                float fpx = 0;
                float fpy = 0;
                float fvx = 0;
                float fvy = 0;

                for (j = 0; j < n; j++)
                {
                    // check for kernel proximity
                    if (dist(i, j) <= h)
                    {
                        Point p = dw(px[i] - px[j], py[i] - py[j]);
                        fpx -= mass[j] * (pressure[i] + pressure[j]) / (2 * density[j]) * p.x;
                        fpy -= mass[j] * (pressure[i] + pressure[j]) / (2 * density[j]) * p.y;

                        fvx -= my * mass[j] * (vx[j] - vx[i]) / density[j] * ddw(px[i] - px[j], py[i] - py[j]);
                        fvy -= my * mass[j] * (vy[j] - vy[i]) / density[j] * ddw(px[i] - px[j], py[i] - py[j]);
                    }
                }

                fx[i] = fpx + fvx;
                fy[i] = fpy + fvy + g;

                if (px[i] < xmin)
                { // outside WEST
                    fy[i] -= signum(fy[i]) * min(f * Math.Abs(fx[i]), Math.Abs(fy[i]));
                    fx[i] = (xmin - px[i] - vx[i] * dt) / (dt * dt);
                }
                else if (xmax < px[i])
                { // outside EAST
                    fy[i] -= signum(fy[i]) * min(f * Math.Abs(fx[i]), Math.Abs(fy[i]));
                    fx[i] = (xmax - px[i] - vx[i] * dt) / (dt * dt);
                }
                else if (py[i] < ymin)
                { // outside SOUTH
                    fx[i] -= signum(fx[i]) * min(f * Math.Abs(fy[i]), Math.Abs(fx[i]));
                    fy[i] = (ymin - py[i] - vy[i] * dt) / (dt * dt);
                }
                else if (xmax < px[i])
                { // outside NORTH
                    fx[i] -= signum(fx[i]) * min(f * Math.Abs(fy[i]), Math.Abs(fx[i]));
                    fy[i] = (ymax - py[i] - vy[i] * dt) / (dt * dt);
                }

                // do euler steps
                vx[i] += fx[i] * dt;
                vy[i] += fy[i] * dt;

                px[i] += vx[i] * dt;
                py[i] += vy[i] * dt;
            }
            //    DrawScreen(screen);
            // Sets the color that the screen will be cleared with.
            SDL_SetRenderDrawColor(renderer, 135, 206, 235, 255);

            // Clears the current render surface.
            SDL_RenderClear(renderer);

            for (i = 0; i < n; i++)
            {
                int x = (int)(px[i] * WIDTH);
                int y = (int)(py[i] * HEIGHT);

                if (x < 1 || WIDTH < x || y < 1 || HEIGHT < y)
                    continue;

                draw_circle(x, y, (int)(h * WIDTH), 254, 254, 254);
            }

            // Switches out the currently presented render surface with the one we just did work on.
            SDL_RenderPresent(renderer);



            //    // Uncomment to save screenshots of every 10th frame.
            //    //if(iter%10==0){
            //    //  char *a = malloc(sizeof(char)*100);
            //    //  sprintf(a, "tmp/FILE%05d.BMP", frame++);
            //    //  SDL_SaveBMP(screen, a);
            //    //}
            //  }

        }

        static void PollEvents()
        {
            // Check to see if there are any events and continue to do so until the queue is empty.
            while (SDL_PollEvent(out SDL_Event e) == 1)
            {
                switch (e.type)
                {
                    case SDL_EventType.SDL_QUIT:
                        running = false;
                        break;
                }
            }
        }

        static void Setup()
        {
            // Initilizes SDL.
            if (SDL_Init(SDL_INIT_VIDEO) < 0)
            {
                Console.WriteLine($"There was an issue initializing SDL. {SDL_GetError()}");
            }

            // Create a new window given a title, size, and passes it a flag indicating it should be shown.
            window = SDL_CreateWindow(
                "SDL .NET 6 Tutorial",
                SDL_WINDOWPOS_UNDEFINED,
                SDL_WINDOWPOS_UNDEFINED,
                WIDTH,
                HEIGHT,
                SDL_WindowFlags.SDL_WINDOW_SHOWN | SDL_WindowFlags.SDL_WINDOW_OPENGL);

            if (window == IntPtr.Zero)
            {
                Console.WriteLine($"There was an issue creating the window. {SDL_GetError()}");
            }

            // Creates a new SDL hardware renderer using the default graphics device with VSYNC enabled.
            renderer = SDL_CreateRenderer(
                window,
                -1,
                SDL_RendererFlags.SDL_RENDERER_ACCELERATED |
                SDL_RendererFlags.SDL_RENDERER_PRESENTVSYNC);

            if (renderer == IntPtr.Zero)
            {
                Console.WriteLine($"There was an issue creating the renderer. {SDL_GetError()}");
            }
        }

        /// <summary>
        /// Renders to the window.
        /// </summary>
        static void Render()
        {
            // Sets the color that the screen will be cleared with.
            SDL_SetRenderDrawColor(renderer, 135, 206, 235, 255);

            // Clears the current render surface.
            SDL_RenderClear(renderer);

            // Set the color to red before drawing our shape
            SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);

            // Draw a line from top left to bottom right
            SDL_RenderDrawLine(renderer, 0, 0, 640, 480);

            // Draws a point at (20, 20) using the currently set color.
            SDL_RenderDrawPoint(renderer, 20, 20);

            // Specify the coordinates for our rectangle we will be drawing.
            var rect = new SDL_Rect
            {
                x = 300,
                y = 100,
                w = 50,
                h = 50
            };

            // Draw a filled in rectangle.
            SDL_RenderFillRect(renderer, ref rect);

            // Switches out the currently presented render surface with the one we just did work on.
            SDL_RenderPresent(renderer);
        }

        static void draw_circle(int n_cx, int n_cy, int radius, byte r, byte g, byte b)
        {
            // invert
            n_cx = WIDTH - n_cx;
            n_cy = HEIGHT - n_cy;

            // if the first pixel in the screen is represented by (0,0) (which is in sdl)
            // remember that the beginning of the circle is not in the middle of the pixel
            // but to the left-top from it:

            double error = (double)-radius;
            double x = (double)radius - 0.5;
            double y = (double)0.5;
            double cx = n_cx - 0.5;
            double cy = n_cy - 0.5;
            //Uint32 color = SDL_MapRGB(surface->format, r, g, b);

            SDL_SetRenderDrawColor(renderer, r, g, b, 255);

            while (x >= y)
            {
                SDL_RenderDrawPoint(renderer, (int)(cx + x), (int)(cy + y));
                SDL_RenderDrawPoint(renderer, (int)(cx + y), (int)(cy + x));

                if (x != 0)
                {
                    SDL_RenderDrawPoint(renderer, (int)(cx - x), (int)(cy + y));
                    SDL_RenderDrawPoint(renderer, (int)(cx + y), (int)(cy - x));
                }

                if (y != 0)
                {
                    SDL_RenderDrawPoint(renderer, (int)(cx + x), (int)(cy - y));
                    SDL_RenderDrawPoint(renderer, (int)(cx - y), (int)(cy + x));
                }

                if (x != 0 && y != 0)
                {
                    SDL_RenderDrawPoint(renderer, (int)(cx - x), (int)(cy - y));
                    SDL_RenderDrawPoint(renderer, (int)(cx - y), (int)(cy - x));
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

        /// <summary>
        /// Clean up the resources that were created.
        /// </summary>
        static void CleanUp()
        {
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            SDL_Quit();
        }
    }
}
