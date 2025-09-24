#!/usr/bin/perl -w

use strict;
use POSIX;

my $lmax = 2.25;
my $lmin = -1.0;
my $bmax = -0.25;
my $bmin = -2.5;

#my $mmin=log10(31.7828133);
#my $mmax=log10(3178.28133);
my $mmin=log10(0.1);
my $mmax=log10(10000.);


#IDRM (interum Design Reference Mission) fields
my $runname = "pearl_test_uniform_draw";
#my @rundes=("kgrizuniform");
#jsut a descriptive name, this unique from previous descriptions
my @rundes=("");
#rl is number of simulated events in a subrun (these are sequential)
my @rl=(10000);
#nr is number of subruns (these parallelize)
my @nr=(1);


#max min semimajor axis
my $amin=log10(0.3);
my $amax=log10(30);

#inclination, period, semimajor, mass
my $inc; my $p; my $a; my $mass;
#pfile is filename produced
my $pfile;

#just pi
my $pi=4.0*atan(1);

#figre out later
my $nl=$rl[0];
my $nf=$nr[0];

my $use_covfac=0;

#for r less than length of rundes
for (my $r=0; $r<@rundes; $r++)
{
    #get the mass from the rundes
    #my $mm=int($rundes[$r])/10.0;
    my $nf = $nr[$r];
    my $nl = $rl[$r];
    my $dir = "${runname}${rundes[$r]}";
    #make dir with rundes as name in directory you run from
    if(! -d $dir)
    {
        mkdir($dir);
    }
    #XXX probably need to check
    if($use_covfac){
	open(FLD,"") || die "Could not open covfac file\n";
    }
    else{
	open(FLD,"./gulls_surot2d_H2023.sources") || die "Could not open sources file\n";
    }

    #rundes.sightline.subrun or some combo
    #looping line over list of sightlines or fields
    
    #perl function to read succesive lines from file
    while(my $line=<FLD>)
    {
	#remvoes eol
	chomp($line);
	my $f=0;
	if($use_covfac){
	    #split lines by spaces into array
	    my @data = split(' ',$line);
	    #if length of aray is greater than7, things didnt split right or not right number of columns
	    if(@data!=2){next;}
	    #make f equal to first element of array (field number)
	    $f = $data[0];
	    # get the (l,b) center of the field
	    my $cov = $data[1];
	    my $fno = $data[0];
	    # check if in bounds
	    if($cov<0.0000001){print "Zero covfac for $f\n";next;}
	}
	else{
	    #split lines by spaces into array
	    my @data = split(' ',$line);
	    #if length of aray is greater than7, things didnt split right or not right number of columns
	    if(@data!=7){next;}
	    #make f equal to first element of array (field number)
	    my $f = $data[0];
	    # get the (l,b) center of the field
	    my $ldeg = $data[1];
	    my $bdeg = $data[2];
	    # check if in bounds
	    if($ldeg>$lmax){next;}
	    if($ldeg<$lmin){next;}
	    if($bdeg>$bmax){next;}
	    if($bdeg<$bmin){next;}
	}

	
	#gen filename base
	my $base = "$dir/${runname}${rundes[$r]}.planets.$f.";
	
	print "On $base: writing $nf files with $nl lines each \n";

	#loop over number of subruns
	for(my $i=0;$i<$nf;$i++)
	{
	    
	    $pfile = "$base$i";
	    #if file exists, don't bother generating
	    if( -e $pfile){next;}
	    #other open up and write to file
	    open(OUT,">$pfile") || die "Could not open output file $pfile\n";
	    
	    #now loop over the number of events generated
	    for(my $j=0;$j<$nl;$j++)
	    {
		$a = 10**($amin + ($amax-$amin)*rand());
		$mass = 3.00374072e-6*10**($mmin + ($mmax-$mmin)*rand());;
		#$a = 1+int(3*rand())/2;
		my $rnd = rand();
		$inc = 180*($rnd<0.5?acos(2*$rnd):-acos(2-2*$rnd))/$pi;
		#$inc = -90.0 + 180.0*rand();
		$p = 360.0*rand();
		printf OUT "%.8e %0.6f %0.6f %0.6f\n", $mass, $a, $inc, $p;
	    } #end for nlines

	    close(OUT);
	} #end for nfiles
    } #end for fields
    close(FLD);
} #end for rundes

