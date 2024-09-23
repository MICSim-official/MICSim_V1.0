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
#include "Multiplier.h"

using namespace std;

Multiplier::Multiplier(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), adder(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}

void Multiplier::Initialize(int _numBit, int _numMultiplier, int _numReadPulse, double _clkFreq) {
	if (initialized)
		cout << "[Multiplier] Warning: Already initialized!" << endl;
	
	numBit = _numBit;                      
	numMultiplier = _numMultiplier;                  
	numReadPulse = _numReadPulse;
	clkFreq = _clkFreq;

	adder.Initialize(numBit, numBit-1, clkFreq);   // N-bit Multiplier needs (N-1)* N-bit Adders

	widthNandN = 2 * MIN_NMOS_SIZE * tech.featureSize;
	widthNandP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	widthNandN2 = 4 * MIN_NMOS_SIZE * tech.featureSize;
	widthNandP2 = 2 * tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	widthInvN = MIN_NMOS_SIZE * tech.featureSize;
	widthInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	widthInvN2 = 2 * MIN_NMOS_SIZE * tech.featureSize;
	widthInvP2 = 2 * tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	
	initialized = true;
}

void Multiplier::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[Multiplier] Error: Require initialization first!" << endl;
	} else {
		double hInv, wInv, hNand, wNand, hInv2, wInv2, hNand2, wNand2;
		// NAND2
		CalculateGateArea(NAND, 2, widthNandN, widthNandP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNand, &wNand);
		// INV
		CalculateGateArea(INV, 1, widthInvN, widthInvP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hInv, &wInv);
		// NAND2_2
		CalculateGateArea(NAND, 2, widthNandN2, widthNandP2, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNand2, &wNand2);
		// INV_2
		CalculateGateArea(INV, 1, widthInvN2, widthInvP2, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hInv2, &wInv2);
		
		double hAdder = hNand+hInv;
		double wAdder = max(wNand, wInv) * numBit;

		adder.CalculateArea(hAdder, wAdder, MAGIC);
		
		area = ((/*(hNand*wNand + hInv*wInv)*numBit*0.5 */ + (hNand2*wNand2 + hInv2*wInv2)*numBit)*numBit + 1.2*adder.area*(numBit-1))*numMultiplier;
		width = _newWidth;
		height = area/width;
		
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

		// NAND2 capacitance
		CalculateGateCapacitance(NAND, 2, widthNandN, widthNandP, hNand, tech, &capNandInput, &capNandOutput);
		// INV capacitance
		CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech, &capInvInput, &capInvOutput);
		// NAND22 capacitance
		CalculateGateCapacitance(NAND, 2, widthNandN2, widthNandP2, hNand, tech, &capNandInput2, &capNandOutput2);
		// INV2 capacitance
		CalculateGateCapacitance(INV, 1, widthInvN2, widthInvP2, hInv, tech, &capInvInput2, &capInvOutput2);
		
	}
}

void Multiplier::CalculateLatency(double _rampInput, double _capLoad, double numRead) {
	if (!initialized) {
		cout << "[Multiplier] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		rampInput = _rampInput;
		capLoad = _capLoad;
		double tr;		/* time constant */
		double gm;		/* transconductance */
		double beta;	/* for horowitz calculation */
		double resPullUp, resPullDown;
		double readLatencyIntermediate = 0;
		double ramp[20];
		double readLatency_1 = 0;
		double readLatency_2 = 0;
		
		ramp[0] = rampInput;

		// Nand to Inv
		resPullDown = CalculateOnResistance(widthNandN, NMOS, inputParameter.temperature, tech);
		tr = resPullDown * (capNandOutput + capInvInput);
		gm = CalculateTransconductance(widthNandN, NMOS, tech);
		beta = 1 / (resPullDown * gm);
		readLatency_1 += horowitz(tr, beta, ramp[0], &ramp[1]);

		// Adder Part
		// 1st
		resPullDown = CalculateOnResistance(widthNandN, NMOS, inputParameter.temperature, tech) * 2;
		tr = resPullDown * (capNandOutput + capNandInput * 3);
		gm = CalculateTransconductance(widthNandN, NMOS, tech);
		beta = 1 / (resPullDown * gm);
		readLatency_1 += horowitz(tr, beta, ramp[1], &ramp[2]);
		
		// 2nd
		resPullUp = CalculateOnResistance(widthNandP, PMOS, inputParameter.temperature, tech);
		tr = resPullUp * (capNandOutput + capNandInput * 2);
		gm = CalculateTransconductance(widthNandP, PMOS, tech);
		beta = 1 / (resPullUp * gm);
		readLatency_1 += horowitz(tr, beta, ramp[2], &ramp[3]);
		
		// 3rd
		resPullDown = CalculateOnResistance(widthNandN, NMOS, inputParameter.temperature, tech) * 2;
		tr = resPullDown * (capNandOutput + capNandInput * 3);
		gm = CalculateTransconductance(widthNandN, NMOS, tech);
		beta = 1 / (resPullDown * gm);
		readLatencyIntermediate += horowitz(tr, beta, ramp[3], &ramp[4]);

		// 4th
		resPullUp = CalculateOnResistance(widthNandP, PMOS, inputParameter.temperature, tech);
		tr = resPullUp * (capNandOutput + capNandInput * 2);
		gm = CalculateTransconductance(widthNandP, PMOS, tech);
		beta = 1 / (resPullUp * gm);
		readLatencyIntermediate += horowitz(tr, beta, ramp[4], &ramp[5]);
		
		if (numBit > 2) {
			readLatency_1 += readLatencyIntermediate * (numBit - 2);
		}
		
		// 5th
		resPullDown = CalculateOnResistance(widthNandN, NMOS, inputParameter.temperature, tech) * 2;
		tr = resPullDown * (capNandOutput + capNandInput * 3);
		gm = CalculateTransconductance(widthNandN, NMOS, tech);
		beta = 1 / (resPullDown * gm);
		readLatency_1 += horowitz(tr, beta, ramp[5], &ramp[6]);

		// 6th
		resPullUp = CalculateOnResistance(widthNandP, PMOS, inputParameter.temperature, tech);
		tr = resPullUp * (capNandOutput + capNandInput);
		gm = CalculateTransconductance(widthNandP, PMOS, tech);
		beta = 1 / (resPullUp * gm);
		readLatency_1 += horowitz(tr, beta, ramp[6], &ramp[7]);
		
		// 7th
		resPullDown = CalculateOnResistance(widthNandN, NMOS, inputParameter.temperature, tech) * 2;
		tr = resPullDown * (capNandOutput + capLoad);
		gm = CalculateTransconductance(widthNandN, NMOS, tech);
		beta = 1 / (resPullDown * gm);
		readLatency_1 += horowitz(tr, beta, ramp[7], &ramp[8]);

		
		if (numBit > 2) {

			// Nand2 to Inv2
			resPullDown = CalculateOnResistance(widthNandN2, NMOS, inputParameter.temperature, tech);
			tr = resPullDown * (capNandOutput2 + capInvInput2);
			gm = CalculateTransconductance(widthNandN2, NMOS, tech);
			beta = 1 / (resPullDown * gm);
			readLatency_2 += horowitz(tr, beta, ramp[8], &ramp[9]);
			
			// Adder Part
			// 1st
			resPullDown = CalculateOnResistance(widthNandN2, NMOS, inputParameter.temperature, tech) * 2;
			tr = resPullDown * (capNandOutput2 + capNandInput2 * 3);
			gm = CalculateTransconductance(widthNandN2, NMOS, tech);
			beta = 1 / (resPullDown * gm);
			readLatency_2 += horowitz(tr, beta, ramp[9], &ramp[10]);
			
			// 2nd
			resPullUp = CalculateOnResistance(widthNandP2, PMOS, inputParameter.temperature, tech);
			tr = resPullUp * (capNandOutput2 + capNandInput2 * 2);
			gm = CalculateTransconductance(widthNandP2, PMOS, tech);
			beta = 1 / (resPullUp * gm);
			readLatency_2 += horowitz(tr, beta, ramp[10], &ramp[11]);
			
			// 3rd
			resPullDown = CalculateOnResistance(widthNandN2, NMOS, inputParameter.temperature, tech) * 2;
			tr = resPullDown * (capNandOutput2 + capNandInput2 * 3);
			gm = CalculateTransconductance(widthNandN2, NMOS, tech);
			beta = 1 / (resPullDown * gm);
			readLatencyIntermediate += horowitz(tr, beta, ramp[11], &ramp[12]);

			// 4th
			resPullUp = CalculateOnResistance(widthNandP2, PMOS, inputParameter.temperature, tech);
			tr = resPullUp * (capNandOutput2 + capNandInput2 * 2);
			gm = CalculateTransconductance(widthNandP2, PMOS, tech);
			beta = 1 / (resPullUp * gm);
			readLatencyIntermediate += horowitz(tr, beta, ramp[12], &ramp[13]);
			
			if (numBit > 2) {
				readLatency_2 += readLatencyIntermediate * (numBit - 2);
			}
			
			// 5th
			resPullDown = CalculateOnResistance(widthNandN2, NMOS, inputParameter.temperature, tech) * 2;
			tr = resPullDown * (capNandOutput2 + capNandInput2 * 3);
			gm = CalculateTransconductance(widthNandN2, NMOS, tech);
			beta = 1 / (resPullDown * gm);
			readLatency_2 += horowitz(tr, beta, ramp[13], &ramp[14]);

			// 6th
			resPullUp = CalculateOnResistance(widthNandP2, PMOS, inputParameter.temperature, tech);
			tr = resPullUp * (capNandOutput2 + capNandInput2);
			gm = CalculateTransconductance(widthNandP2, PMOS, tech);
			beta = 1 / (resPullUp * gm);
			readLatency_2 += horowitz(tr, beta, ramp[14], &ramp[15]);
			
			// 7th
			resPullDown = CalculateOnResistance(widthNandN2, NMOS, inputParameter.temperature, tech) * 2;
			tr = resPullDown * (capNandOutput2 + capLoad);
			gm = CalculateTransconductance(widthNandN2, NMOS, tech);
			beta = 1 / (resPullDown * gm);
			readLatency_2 += horowitz(tr, beta, ramp[15], &ramp[16]);
		
		}
		
		
		if (numBit > 2) {			
			readLatency = readLatency_1*0.5*(numBit-1) + readLatency_2*0.5*(numBit-1);	
			rampOutput = ramp[16];
		} else {		
			readLatency = readLatency_1*numBit;
			rampOutput = ramp[8];
		}
		
        readLatency *= numRead;		
	}
}

void Multiplier::CalculatePower(double numRead) {
	if (!initialized) {
		cout << "[Multiplier] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		double readDynamicEnergy_1 = 0;
		double readDynamicEnergy_2 = 0;
		
		/* Leakage power */
		leakage += CalculateGateLeakage(NAND, 2, widthNandN, widthNandP, inputParameter.temperature, tech) * tech.vdd * 9 * numBit * (numBit-1) *0.5;
		leakage += CalculateGateLeakage(NAND, 2, widthNandN2, widthNandP2, inputParameter.temperature, tech) * tech.vdd * 9 * numBit * (numBit-1) *0.5;
		leakage *= numMultiplier;

		
		// (0.5*N*N)* Nand to Inv
		readDynamicEnergy_1 += numBit*(numBit*capNandInput) * tech.vdd * tech.vdd;    
        readDynamicEnergy_1 += numBit*(numBit*capNandOutput+numBit*capInvInput) * tech.vdd * tech.vdd;  

		/* Read Dynamic energy */   /***Adder part***/
		// Calibration data pattern of critical path is A=1111111..., B=1000000... and Cin=1
		// Only count 0 to 1 transition for energy
		// First stage
		readDynamicEnergy_1 += (capNandInput * 6) * tech.vdd * tech.vdd;    // Input of 1 and 2 and Cin
        readDynamicEnergy_1 += (capNandOutput * 2) * tech.vdd * tech.vdd;  // Output of S[0] and 5
		// Second and later stages
		readDynamicEnergy_1 += (capNandInput * 7) * tech.vdd * tech.vdd * (numBit-1);
		readDynamicEnergy_1 += (capNandOutput * 3) * tech.vdd * tech.vdd * (numBit-1);
		
		// Hidden transition
		// First stage
		readDynamicEnergy_1 += (capNandOutput + capNandInput) * tech.vdd * tech.vdd * 2;	// #2 and #3
		readDynamicEnergy_1 += (capNandOutput + capNandInput * 2) * tech.vdd * tech.vdd;	// #4
		readDynamicEnergy_1 += (capNandOutput + capNandInput * 3) * tech.vdd * tech.vdd;	// #5
		readDynamicEnergy_1 += (capNandOutput + capNandInput) * tech.vdd * tech.vdd;		// #6
		// Second and later stages
		readDynamicEnergy_1 += (capNandOutput + capNandInput * 3) * tech.vdd * tech.vdd * (numBit-1);	// # 1
		readDynamicEnergy_1 += (capNandOutput + capNandInput) * tech.vdd * tech.vdd * (numBit-1);		// # 3
		readDynamicEnergy_1 += (capNandOutput + capNandInput) * tech.vdd * tech.vdd * 2 * (numBit-1);		// #6 and #7
		
		if (numBit > 2) {
			
			// (0.5*N*N)* Nand2 to Inv2
			readDynamicEnergy_2 += numBit*(numBit*capNandInput2) * tech.vdd * tech.vdd;    
			readDynamicEnergy_2 += numBit*(numBit*capNandOutput2+numBit*capInvInput2) * tech.vdd * tech.vdd;  
			
			/* Read Dynamic energy */   /***Adder part***/
			// Calibration data pattern of critical path is A=1111111..., B=1000000... and Cin=1
			// Only count 0 to 1 transition for energy
			// First stage
			readDynamicEnergy_2 += (capNandInput2 * 6) * tech.vdd * tech.vdd;    // Input of 1 and 2 and Cin
			readDynamicEnergy_2 += (capNandOutput2 * 2) * tech.vdd * tech.vdd;  // Output of S[0] and 5
			// Second and later stages
			readDynamicEnergy_2 += (capNandInput2 * 7) * tech.vdd * tech.vdd * (numBit-1);
			readDynamicEnergy_2 += (capNandOutput2 * 3) * tech.vdd * tech.vdd * (numBit-1);
			
			// Hidden transition
			// First stage
			readDynamicEnergy_2 += (capNandOutput2 + capNandInput2) * tech.vdd * tech.vdd * 2;	// #2 and #3
			readDynamicEnergy_2 += (capNandOutput2 + capNandInput2 * 2) * tech.vdd * tech.vdd;	// #4
			readDynamicEnergy_2 += (capNandOutput2 + capNandInput2 * 3) * tech.vdd * tech.vdd;	// #5
			readDynamicEnergy_2 += (capNandOutput2 + capNandInput2) * tech.vdd * tech.vdd;		// #6
			// Second and later stages
			readDynamicEnergy_2 += (capNandOutput2 + capNandInput2 * 3) * tech.vdd * tech.vdd * (numBit-1);	// # 1
			readDynamicEnergy_2 += (capNandOutput2 + capNandInput2) * tech.vdd * tech.vdd * (numBit-1);		// # 3
			readDynamicEnergy_2 += (capNandOutput2 + capNandInput2) * tech.vdd * tech.vdd * 2 * (numBit-1);		// #6 and #7
			
		}
		
		
		if (numBit > 2) {			
			readDynamicEnergy = readDynamicEnergy_1 * 0.5*(numBit-1) + readDynamicEnergy_2 * 0.5*(numBit-1);			
		} else {		
			readDynamicEnergy = readDynamicEnergy_1 *(numBit-1);			
		}
	
		readDynamicEnergy *= numMultiplier;	


		// if (!readLatency) {
			// cout << "[Multiplier] Error: Need to calculate read latency first" << endl;
		// } else {
			// readPower = readDynamicEnergy/readLatency;
		// }
	}
}

void Multiplier::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

