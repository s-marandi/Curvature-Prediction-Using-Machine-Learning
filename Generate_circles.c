#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "tension.h"
#include "reduced.h"
#include <stdio.h>


int gminlevel = 6;
int gmaxlevel = 9;
double finaltime = 1.00;
double R0;
int cnt = 0;

int main() {
/* Move origin to (X0,Y0). Default is (0,0) */
    L0 = 1;
    X0 = -L0/2;
    Y0 = -L0/2;

// Phase 1
    f.sigma = 10.;
    mu1 = .5;
    rho1 = 50.;
// Phase 1
    mu2 = .5;
    rho2 = 1.;

    N = 1 << gminlevel; // Number of grid points pow(2,gminlevel)
    

FILE *myFile;
myFile = fopen("radii.txt", "r");

double numbers[65];
int i;

for (i = 0; i < 65; i++)
{
    fscanf(myFile, "%f,", &numbers[i]);
	R0 = numbers[i];
	run();
        cnt += 1;
}
    }


event logfile (i++)
{
	printf("i=%d, t=%g\n",i,t);
}

event init(t=0) {
    /* refine (sq(x) + sq(y) < 1.1 && level < gmaxlevel); */
// Static refinement upto gmaxlelvel and 1.1 times the circle radius
/* Initialize a circle of radius R0 */
    fraction (f, sq(R0) - sq(x) - sq (y)); // The drop
/* The function fills c with vertex scalar field Phi */ 
/* struct Fractions { */
/* 	vertex scalar Phi; // compulsory */
/* 	scalar c;          // compulsory */
/* 	face vector s;     // optional */
/* 	double val;        // optional (default zero) */
/* }; */
    DT = 0.001;
}

event qoi (i = 10) {
    scalar kappa[], radius[];
    char fname[30];
    sprintf(fname, "R0_%03d.txt", cnt);
    FILE *fp = fopen(fname, "w");

    fprintf (fp, "# hk f[i-1,j+1] f[i-1,j] f[i-1,j-1] " 
                      "f[i,j+1] f[i,j] f[i,j-1] "  
                      "f[i+1,j+1] f[i+1,j] f[i+1,j-1]\n");

    /* cstats s = curvature (f, kappa); */
    curvature (f, kappa);
    foreach(){
        /* Extract interface f values in (0,1) */
        if (f[] > 0. && f[] < 1.){
            fprintf (fp, "%7.6f %7.6f %7.6f %7.6f %7.6f %7.6f %7.6f %7.6f %7.6f %7.6f\n",
                     Delta/R0, f[], f[-1,0], f[1,0], f[0,-1], f[0,1], 
                     f[-1,-1], f[1,-1], f[-1,1], f[1,1]);
        }
    }
    fclose(fp);
}

/* End simulation after 10 timesteps */
event stopcode (i = 10) {
    return 1;
}