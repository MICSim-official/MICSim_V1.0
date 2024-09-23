/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
* 
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
* 
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
* 
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen	    Email: pchen72 at asu dot edu 
*                    
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cmath>
#include <iostream>
#include "constant.h"
#include "formula.h"
#include "RecfgAdderTree.h"

using namespace std;

RecfgAdderTree::RecfgAdderTree(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), 
								adder(_inputParameter, _tech, _cell), demux(_inputParameter, _tech, _cell), mux(_inputParameter, _tech, _cell), muxDecoder(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}

void RecfgAdderTree::Initialize(int _numSubcoreRow, int _numAdderBit, int _numAdderTree) {
	if (initialized)
		cout << "[RecfgAdderTree] Warning: Already initialized!" << endl;
	
	numSubcoreRow = _numSubcoreRow;                  // # of row of subcore in the synaptic core
	numStage = ceil(log2(numSubcoreRow));            // # of stage of the adder tree, used for CalculateLatency ...
	numAdderBit = _numAdderBit;                      // # of input bits of the Adder
	numAdderTree = _numAdderTree;                    // # of Adder Tree
	
	initialized = true;
}

void RecfgAdderTree::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[RecfgAdderTree] Error: Require initialization first!" << endl;
	} else {
		double hInv, wInv, hNand, wNand;
		
		// Adder
		int numAdderEachStage = 0;                          // define # of adder in each stage
		int numBitEachStage = numAdderBit;                  // define # of bits of the adder in each stage
		int numAdderEachTree = 0;                           // define # of Adder in each Adder Tree
		int nummuxDecoderEachTree = 0;
		int i = ceil(log2(numSubcoreRow));
		int j = numSubcoreRow;
		
		while (i != 0) {  // calculate the total # of full adder in each Adder Tree
			numAdderEachStage = ceil(j/2);
			numAdderEachTree += numBitEachStage*numAdderEachStage;
			nummuxDecoderEachTree += numAdderEachStage;
			numBitEachStage += 1;
			j = ceil(j/2);
			i -= 1;
		}
		adder.Initialize(numAdderEachTree, numAdderTree, 1e9);
		demux.Initialize(2*numAdderEachTree*numAdderTree, 2, 0, true); //numInput = 2*numAdderEachTree; numSelection = 2; num=numAdderTree
		mux.Initialize(1*numAdderEachTree*numAdderTree, 3, 0, true); //numInput = numAdderEachTree; numSelection = 3; num=numAdderTree
		muxDecoder.Initialize((_newWidth? REGULAR_COL:REGULAR_ROW), 2, false, false); //numAddrRow = 2;
		
		if (_newWidth && _option==NONE) {
			adder.CalculateArea(NULL, _newWidth, NONE);
			demux.CalculateArea(NULL, _newWidth, NONE);
			mux.CalculateArea(NULL, _newWidth, NONE);
			muxDecoder.CalculateArea(NULL, NULL, NONE);
			width = _newWidth;
			height = (adder.area + (demux.area+mux.area+muxDecoder.area*nummuxDecoderEachTree*numAdderTree)) / width;
		} else if (_newHeight && _option==NONE) {
			adder.CalculateArea(_newHeight, NULL, NONE);
			demux.CalculateArea(_newHeight, NULL, NONE);
			mux.CalculateArea(_newHeight, NULL, NONE);
			muxDecoder.CalculateArea(NULL, NULL, NONE);
			height = _newHeight;
			width = (adder.area + (demux.area+mux.area+muxDecoder.area*nummuxDecoderEachTree*numAdderTree)) / height;
		} else {
			cout << "[RecfgAdderTree] Error: No width assigned for the adder tree circuit" << endl;
			exit(-1);
		}
		// cout<<"adder.area: "<<adder.area<<endl;
		// // cout<<"nummuxDecoderEachTree*numAdderTree: "<<nummuxDecoderEachTree*numAdderTree<<endl;
		// cout<<"demux.area: "<<demux.area<<endl;
		// cout<<"mux.area: "<<mux.area<<endl;
		// cout<<"muxDecoder.area: "<<muxDecoder.area<<endl;
		// cout<<"muxDecoder.area*nummuxDecoderEachTree*numAdderTree: "<<muxDecoder.area*nummuxDecoderEachTree*numAdderTree<<endl;
		area = height*width;
		adder.initialized = false;
		demux.initialized = false;
		mux.initialized = false;

		// Modify layout
		newHeight = _newHeight;
		newWidth = _newWidth;
		switch (_option) {
			case MAGIC:
				MagicLayout();
				break;
			case OVERRIDE:
				OverrideLayout();
				break;  
			default:    // NONE
				break;
		}

	}
}

void RecfgAdderTree::CalculateLatency(double numRead, int numUnitAdd, int numBitAdd, double _capLoad) {
	if (!initialized) {
		cout << "[RecfgAdderTree] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		
		int numAdderEachStage = 0;                          // define # of adder in each stage
		int numBitEachStage = min(numBitAdd, numAdderBit);                  // define # of bits of the adder in each stage
		int numAdderEachTree = 0;                           // define # of Adder in each Adder Tree
		int i = 0;
		int j = 0;
		
		if (!numUnitAdd) {
			i = ceil(log2(numSubcoreRow));
			j = numSubcoreRow;
		} else {
			i = ceil(log2(numUnitAdd));
			j = numUnitAdd;
		}
		int numBypassStage = numStage-i;
		
		while (i != 0) {   // calculate the total # of full adder in each Adder Tree
			numAdderEachStage = ceil(j/2);
			//demux
			demux.Initialize(2*numBitEachStage, 2, 0, true); //numInput = 2*numBitEachStage; numSelection = 2; num=numAdderTree
			demux.CalculateLatency(0,_capLoad,1);
			//adder
			adder.Initialize(numBitEachStage, numAdderEachStage, 1e9);
			adder.CalculateLatency(1e20, _capLoad, 1);				
			numBitEachStage += 1;
			//mux
			mux.Initialize(numBitEachStage, 3, 0, true); //numInput = numBitEachStage; numSelection = 3; num=numAdderTree
			mux.CalculateLatency(0,_capLoad,1);
			//muxDecoder: every stage at the same time
			muxDecoder.CalculateLatency(1e20, mux.capTgGateN*numBitEachStage, mux.capTgGateP*numBitEachStage, 1, 0);
			
			readLatency += (adder.readLatency+demux.readLatency);
			readLatency = max(readLatency, muxDecoder.readLatency);
			readLatency += mux.readLatency;
			
			adder.initialized = false;
			demux.initialized = false;
			mux.initialized = false;
			
			j = ceil(j/2);
			i -= 1;
		}
		//bypass stages
		demux.Initialize(2*numBitEachStage, 2, 0, true); //numInput = 2*numBitEachStage; numSelection = 2; num=numAdderTree
		demux.CalculateLatency(0,_capLoad,numBypassStage);
		mux.Initialize(1*numBitEachStage, 3, 0, true); //numInput = numBitEachStage; numSelection = 3; num=numAdderTree
		mux.CalculateLatency(0,_capLoad,numBypassStage);
		
		readLatency += (mux.readLatency+demux.readLatency);
		
		demux.initialized = false;
		mux.initialized = false;
		// cout<<"RecfgGaccumulation->readLatency: "<<readLatency<<endl;
        readLatency *= numRead;		
	}
}

void RecfgAdderTree::CalculatePower(double numRead, int numUnitAdd, int numBitAdd, int numusedAdderTree) {
	if (!initialized) {
		cout << "[RecfgAdderTree] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		
		int numAdderEachStage = 0;                          // define # of adder in each stage
		int numBitEachStage = min(numBitAdd, numAdderBit);                // define # of bits of the adder in each stage
		int numAdderEachTree = 0;                           // define # of Adder in each Adder Tree
		int i = 0;
		int j = 0;
		
		if (!numUnitAdd) {
			i = ceil(log2(numSubcoreRow));
			j = numSubcoreRow;
		} else {
			i = ceil(log2(numUnitAdd));
			j = numUnitAdd;
		}
		int numBypassStage = numStage-i;
		
		while (i != 0) {  // calculate the total # of full adder in each Adder Tree
			numAdderEachStage = ceil(j/2);
			//demux
			demux.Initialize(2*numBitEachStage, 2, 0, true); //numInput = 2*numBitEachStage; numSelection = 2; num=numAdderTree
			demux.CalculatePower(numAdderEachStage);
			//adder
			adder.Initialize(numBitEachStage, numAdderEachStage, 1e9);
			adder.CalculatePower(1, numAdderEachStage);	
			numBitEachStage += 1;
			//mux
			mux.Initialize(numBitEachStage, 3, 0, true); //numInput = numBitEachStage; numSelection = 3; num=numAdderTree
			mux.CalculatePower(numAdderEachStage);
			
			readDynamicEnergy += (adder.readDynamicEnergy + demux.readDynamicEnergy + mux.readDynamicEnergy);	
			leakage += adder.leakage;
			
			adder.initialized = false;
			demux.initialized = false;
			mux.initialized = false;
			
			j = ceil(j/2);
			i -= 1;
		}
		//bypass stages
		demux.Initialize(2*numBitEachStage, 2, 0, true); //numInput = 2*numBitEachStage; numSelection = 2; num=numAdderTree
		demux.CalculatePower(numBypassStage);
		mux.Initialize(1*numBitEachStage, 3, 0, true); //numInput = numBitEachStage; numSelection = 3; num=numAdderTree
		mux.CalculatePower(numBypassStage);
		//muxDecoder: every stage
		muxDecoder.CalculatePower(numStage,1);
		
		readDynamicEnergy += (muxDecoder.readDynamicEnergy + demux.readDynamicEnergy + mux.readDynamicEnergy);
		
		demux.initialized = false;
		mux.initialized = false;

		// cout<<"RecfgGaccumulation->readDynamicEnergy: "<<readDynamicEnergy<<endl;
		readDynamicEnergy *= numusedAdderTree;		
		readDynamicEnergy *= numRead;
		leakage *= numusedAdderTree;
	}
}

void RecfgAdderTree::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}