/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    scalarTransportFoam

Group
    grpBasicSolvers

Description
    Passive scalar transport equation solver.

    \heading Solver details
    The equation is given by:

    \f[
        \ddt{T} + \div \left(\vec{U} T\right) - \div \left(D_T \grad T \right)
        = S_{T}
    \f]

    Where:
    \vartable
        T       | Passive scalar
        D_T     | Diffusion coefficient
        S_T     | Source
    \endvartable

    \heading Required fields
    \plaintable
        T       | Passive scalar
        U       | Velocity [m/s]
    \endplaintable

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
// #include "IFstream.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Passive scalar transport equation solver."
    );

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nCalculating scalar transport\n" << endl;

    #include "adjointSettings.H"

    label tapeSizeMB = mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<label>("tapeSizeMB",4096);
    Info << "Creating Tape, size: " << tapeSizeMB << endl;
    AD::createGlobalTape(tapeSizeMB);

    label nCells = T.size();

    std::vector<double> dTdDT1(nCells, 0.0);
    std::vector<double> dTdDT2(nCells, 0.0);
    std::vector<double> dTdU(nCells, 0.0);
    
    // register inputs w.r.t. we want to diff.
    AD::registerInputVariable(regionDT[0]);
    AD::registerInputVariable(regionDT[1]);
    AD::registerInputVariable(Uparam[0]);

    #include "modifyFields.H"
    #include "createPhi.H"
    #include "CourantNo.H"
	
    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix TEqn
            (
                fvm::ddt(T)
              + fvm::div(phi, T)
              - fvm::laplacian(DT, T)
             ==
                fvOptions(T)
            );

            TEqn.relax();
            fvOptions.constrain(TEqn);
            TEqn.solve();
            fvOptions.correct(T);
        }

        runTime.write();
    }

    // Info<< "Time = " << runTime.timeName() << nl << endl;

    // fvScalarMatrix TEqn
    // (
        // fvm::ddt(T)
      // + fvm::div(phi, T)
      // - fvm::laplacian(DT, T)
     // ==
        // fvOptions(T)
    // );

    // TEqn.relax();
    // fvOptions.constrain(TEqn);
    // TEqn.solve();
    // fvOptions.correct(T);

    // runTime.write();
  
    Info<< "Forward execution finished\n" << endl;
    AD::switchTapeToPassive();

    for(int i=0; i<nCells; i++){
		Info << "Interpret " << i << " / " << nCells << endl;
		AD::derivative(T[i]) = 1.0;
		AD::interpretTape();
		dTdDT1[i] = AD::derivative(regionDT[0]);
        dTdDT2[i] = AD::derivative(regionDT[1]);
        dTdU[i] = AD::derivative(Uparam[0]);
		AD::zeroAdjointVector();
	}
	
    Info<< "Reverse execution finished\n" << endl;
    std::ofstream ofs1("jacMu1(T)");
    std::ofstream ofs2("jacMu2(T)");
    std::ofstream ofs3("jacV(T)");
    for(int i=0; i<nCells; i++){
      	ofs1 << dTdDT1[i] << "\n";
        ofs2 << dTdDT2[i] << "\n";
        ofs3 << dTdU[i] << "\n";
	}

    ofs1.close();
    ofs2.close();
    ofs3.close();

    AD::resetTape();

    Foam::profiling::print(Info);
    ADmode::tape_t::remove(ADmode::global_tape);

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
