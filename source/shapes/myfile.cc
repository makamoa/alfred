#include"nanocpp.h"
#include"dft_in_place.h"
#include"tdcmt.h"
using namespace std;


int main(int argc, char* argv[]){

  ////////////////////// INIT THE CODE /////////////////
  nanocpp n(argc,argv);
  cout<<n<<endl; //to screen

  input myInput = input( n.get_inp(), "#" );
  int *npts;
  npts = myInput.get<int>(argv, argc, "number_of_cells", ' ' );
  string rd=n.get_rootdir();

  string myfx,myfz,myfy,myfc,fna,mygeo;
  mygeo = rd+"geometry/material";
  myfx=mygeo+"-x.bin";
  myfy=mygeo+"-y.bin";
  myfz=mygeo+"-z.bin";
  myfc=mygeo+"-c.bin";

  cout << "Importing "<< myfx << endl;
  n.medium.type_Ex.load<MPI_INT>(myfx,n.get_grid());

   cout << "Importing "<< myfy << endl;
  n.medium.type_Ey.load<MPI_INT>(myfy,n.get_grid());

  cout << "Importing "<< myfz << endl;
  n.medium.type_Ez.load<MPI_INT>(myfz,n.get_grid());

  cout << "Importing "<< myfc << endl;
  n.medium.type_Ec.load<MPI_INT>(myfc,n.get_grid());
  cout << "All media files imported" <<  endl;


  n.medium.type_Ex.save<MPI_INT>(rd+"type_Ex.bin",n.get_grid());
  n.medium.type_Ey.save<MPI_INT>(rd+"type_Ey.bin",n.get_grid());
  n.medium.type_Ez.save<MPI_INT>(rd+"type_Ez.bin",n.get_grid());
  n.medium.type_Ec.save<MPI_INT>(rd+"type_Ec.bin",n.get_grid());


  /////////////////// MY CODE ////////////////////
  //init a volume or plane
  //double cds[2*FD_N]={0.5,1.5,0.5,1.5};
  //double cds[2*FD_N]={0.2,1.8,1.5,1.5};
  //int fc[FD_N]={CE,CE};
  //volume<FD_N> kk;
  //kk.init(cds,n.get_grid(),fc);

  ////////////////////////SET INDEX
  // file with material data -> fna
  // the epsilon files are from 300nm to 1200nm, with 1nm of spacing
  // calculate the right position on the array
  fna="mats/Si.bin";    // [wl;er;ei;der] (matlab-like)
  // dimensions of material array
  int dimensions[2]; dimensions[0]=901; dimensions[1]=4;
  //calculate the index of my wavelength (from simulation)
  int myidx=floor(n.get_wl()*1e3-300+0.5);

  // array where I will load my material data
  tensor<2,double> myeps;
  myeps.init(dimensions,0.);
  myeps.load(fna);

  // debug
  //if(n.get_chart().get_me()==0)
    //cout<<setprecision(20)<<myidx<<" "<<myeps(myidx,0)<<" "<<myeps(myidx,1)<<" "<<myeps(myidx,2)<<" "<<myeps(myidx,3)<<endl;
  // set index sets the refractive index of
  n.set_index(2,n.get_w0(),myeps(myidx,1),-myeps(myidx,2));

  ////////////////////////SET INDEX
  // file with material data -> fna
  // the epsilon files are from 300nm to 1200nm, with 1nm of spacing
  // calculate the right position on the array
  //string fna="mats/Si.bin";    // [wl;er;ei;der] (matlab-like)
  // dimensions of material array
  //int dimensions[2]; dimensions[0]=901; dimensions[1]=4;
  //calculate the index of my wavelength (from simulation)
  //int myidx=floor(n.get_wl()*1e3-300+0.5);

  // array where I will load my material data
  //tensor<2,double> myeps;
  //myeps.init(dimensions,0.);
  //myeps.load(fna);

  // debug
  //if(n.get_chart().get_me()==0)
    //cout<<setprecision(20)<<myidx<<" "<<myeps(myidx,0)<<" "<<myeps(myidx,1)<<" "<<myeps(myidx,2)<<" "<<myeps(myidx,3)<<endl;
  // set index sets the refractive index of
  //n.set_index(2,n.get_w0(),myeps(myidx,1),-myeps(myidx,2));
  /////////////////DFT
  //init a volume
  // double cds[2*FD_N]={0.5,1.5,0.5,1.5};
  // int fc[FD_N]={CE,CE};
  // volume<FD_N> vv;
  // vv.init(cds,n.get_grid(),fc);
  // //init a DFT object
  // int dim=3;
  // double *fw=new double[3];
  // fw[0]=2.75;
  // fw[1]=3.25;
  // fw[2]=4.4;
  // for(int i=0;i<dim;i++)
  //   fw[i]*=2*PI*C0;
  // double dt=n.get_dt();
  // dft_vol df(vv,fw,dim,dt);

  // // Poynting flux with harmonic expansion
  // // Init data
  // double cds[4]={0.,0.99,0.8,0.8};
  // int fc[FD_N]={CE,CE};
  // volume<FD_N> vol;
  // vol.init(cds,n.get_grid(),fc);
  // int nmodes=10;

  // // TDCMT
  // int tdim[2]={6,2}; //number of modes they are saved row by row, only even
  // // (x0,z0),(x0,z1),(x0,z2),...

  // //reads data from file
  ifstream fp_in;
  // fp_in.open("sizes.txt", ios::in);
  double tsizz[6];
  // for(int i=0;i<4;i++)
  //   fp_in>>tsizz[i];
  // //double tsizz[4]={0.3,0.7,0.45,0.55}; //size of vol   

  //int tdim[1]={3};
  //tdcmt td;
  //td.setup(tdim,tsizz,n);


  //HARM TRANSM
  int dim[2]={1,1};
  //double sizz[4]={tsizz[0],tsizz[1],tsizz[3],tsizz[3]};
  //double sizz[4]={0.1,0.9,0.8,0.8};
  fp_in.open(n.get_rootdir()+"tfsf.txt", ios::in);
  for(int i=0;i<6;i++)
    fp_in>>tsizz[i];
  double sizz[6]={tsizz[0],tsizz[1],tsizz[2],tsizz[3],tsizz[5],tsizz[5]};
  harm_z ha;
  ha.init(dim,sizz,n);
  //HARM REFL
  sizz[4]=0.25;//0.35;
  sizz[5]=0.25;//0.35;
  harm_z ha_r;
  ha_r.init(dim,sizz,n);
  fp_in.close();

//  ///TOTAL TRANSMISSION
//  volume<FD_N> vol;
//  int fc[FD_N]={CE,CE};
//  vol.init(sizz,n.get_grid(),fc);
//  tensor<2,double> poy;
//  fc[0]=n.total_steps;
//  poy.init(fc,0.);

  /////////////////////////////////////////////////////

  ///////////////////// CYCLE /////////////////////
  double sta=MPI_Wtime();
  for(n.tcurr;n.tcurr<=n.total_steps;n.tcurr++){
    
    n.time=n.tcurr*n.dt;     //update time;
    n.single_step();    //single step
    
    /////////////////// MY CODE ////////////////////
    //n.erg_exch(); //exchange boundaries
    //df.accum_dft_all(n,n.tcurr);// accumulate dft results    

    //TDCMT
    //td.update_A(n);

    //Harm
    ha.update_C(n);        
    ha_r.update_C(n);

//    //FLUX
//    poy[n.tcurr]=n.z_erg_flux(vol);

    /////////////////////////////////////////////////////
  }


  /////////////////// MY CODE ////////////////////
  ////TDCMT
  //string fn=n.get_rootdir()+"/td.bin";
  //td.save(fn);
  //Harm
  string fn1=n.get_rootdir()+"harm0.bin";
  ha.save(fn1);
  fn1=n.get_rootdir()+"harm_r0.bin";
  ha_r.save(fn1);

  //compute erg and poynting in that region
  //double hh=n.total_erg_in_box(kkk);
  //double hh=n.z_erg_flux(kkk);
//  if(vol.get_me()==0){
//  //  cout<<hh<<endl;
//  //  if(kkk.get_me()==0){
//    string fn=n.get_rootdir()+"poy.bin";
//    poy.save(fn);
//  }
  
  //DFT
  // string fn="res/ergw.bin";
  // df.save_erg(n,fn);
  // fn="res";
  // df.save_fields(n,fn);

  // //fn="res/qy.bin";
  // //n.medium.Qy.save(fn);
  //string fn="res/ez.bin";
  //n.Ez.save<MPI_DOUBLE>(fn,n.get_grid());
  //fn="res/ex.bin";
  //n.Ex.save<MPI_DOUBLE>(fn,n.get_grid());
  //fn="res/hy.bin";
  //n.Hy.save<MPI_DOUBLE>(fn,n.get_grid());
  // //df.save_fields(n,fn);
  
  // delete []fw;
  
  /////////////////////////////////////////////////////  
  
  //////////////////// SAVE OUTPUT AND LOGOUT /////////////

  n.post_process();    //process data
  n.goodbye(sta);   //get time and print goodbye
  
  ////////////////////////////////////////////////////////    


}
