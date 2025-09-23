/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019-2021 OpenCFD Ltd.
    Copyright (C) YEAR AUTHOR, AFFILIATION
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

\*---------------------------------------------------------------------------*/

#include "codedFvOptionTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
#include "fvMatrix.H"

//{{{ begin codeInclude

//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace fv
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

// dynamicCode:
// SHA1 = d3a411cc2a3df4852f50f707f966326cbc1756d3
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void codedSource_d3a411cc2a3df4852f50f707f966326cbc1756d3(bool load)
{
    if (load)
    {
        // Code that can be explicitly executed after loading
    }
    else
    {
        // Code that can be explicitly executed before unloading
    }
}


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(codedSourceFvOptionscalarSource, 0);
addRemovableToRunTimeSelectionTable
(
    option,
    codedSourceFvOptionscalarSource,
    dictionary
);

} // End namespace fv
} // End namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::fv::
codedSourceFvOptionscalarSource::
codedSourceFvOptionscalarSource
(
    const word& name,
    const word& modelType,
    const dictionary& dict,
    const fvMesh& mesh
)
:
    fv::cellSetOption(name, modelType, dict, mesh)
{
    if (false)
    {
        printMessage("Construct codedSource fvOption from dictionary");
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::fv::
codedSourceFvOptionscalarSource::
~codedSourceFvOptionscalarSource()
{
    if (false)
    {
        printMessage("Destroy codedSource");
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void
Foam::fv::
codedSourceFvOptionscalarSource::correct
(
    GeometricField<scalar, fvPatchField, volMesh>& fld
)
{
    if (false)
    {
        Info<< "codedSourceFvOptionscalarSource::correct()\n";
    }

//{{{ begin code
    #line 30 "/mnt/c/Users/jgs_j/Desktop/Perth/OpenFOAM/tests/pure_convection/system/fvOptions.codedSource.scalarCodedSourceCoeffs"
//vectorField& testField = fld;
//}}} end code
}


void
Foam::fv::
codedSourceFvOptionscalarSource::addSup
(
    fvMatrix<scalar>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "codedSourceFvOptionscalarSource::addSup()\n";
    }

//{{{ begin code - warn/fatal if not implemented?
    #line 35 "/mnt/c/Users/jgs_j/Desktop/Perth/OpenFOAM/tests/pure_convection/system/fvOptions.codedSource.scalarCodedSourceCoeffs"
const scalarField& V = mesh_.V();
        	scalarField& heSource = eqn.source();
		scalar PI_ = Foam::constant::mathematical::pi;

        	// Retrieve the x and y component of the cell centres
        	//const scalarField& cellx = mesh_.C().component(0);
		//const scalarField& celly = mesh_.C().component(1);

        	// Apply the source
        	//forAll(cellx, i)
        	//{
            	//	// cell volume specific source
            	//	//heSource[i] += 1e5*sin(200*cellx[i])*V[i];
		//	heSource[i] += 1.5*PI_*sin(1.5*PI_*(cellx[i]+celly[i]))*V[i];
          	//}
		
		forAll(mesh_.C(), celli)
		{
			heSource[celli] += 1.5*PI_*sin(1.5*PI_*(mesh_.C()[celli].component(0)+mesh_.C()[celli].component(1)))*V[celli];
		}
//}}} end code
}


void
Foam::fv::
codedSourceFvOptionscalarSource::addSup
(
    const volScalarField& rho,
    fvMatrix<scalar>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "codedSourceFvOptionscalarSource::addSup(rho)\n";
    }

//{{{ begin code - warn/fatal if not implemented?
    NotImplemented
//}}} end code
}


void
Foam::fv::
codedSourceFvOptionscalarSource::constrain
(
    fvMatrix<scalar>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "codedSourceFvOptionscalarSource::constrain()\n";
    }

//{{{ begin code
    
//}}} end code
}


// ************************************************************************* //

