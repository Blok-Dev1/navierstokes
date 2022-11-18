namespace NavierStokes.Core
{
    public struct sim_param_t
    {
        public int n = 700;         /* Number of Frames */
        public double h = 0.015;    /* Particle size    */
        public double dt = 0.002;   /* Time Step */

        public sim_param_t()
        {
        }
        //int plotEach = 1;
        //float xmin = 0;
        //float xmax = 1;
        //float ymin = 0;
        //float ymax = 1;
        //float g = -2;
        //float k = 750;

    }
}