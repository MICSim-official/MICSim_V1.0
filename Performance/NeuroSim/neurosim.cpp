#include <pybind11/pybind11.h>
#include "./funcs.hpp"
#include "./SubArray.h"
#include "./AdderTree.h"
#include "./Bus.h"
#include "./Buffer.h"
#include "./HTree.h"
#include "./BitShifter.h"
#include "./MaxPooling.h"
#include "./typedef.h"
#include "./DRAM.h"
#include "./Mux.h"
#include "./RowDecoder.h"
#include "./Technology.h"
#include "./MultilevelSAEncoder.h"
#include "./Multiplier.h"
#include "./MultilevelSenseAmp.h"
#include "./RecfgAdderTree.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

using namespace pybind11::literals;


PYBIND11_MODULE(neurosim, m)
{
    py::enum_<Type::MemCellType>(m, "MemCellType")
            .value("SRAM", Type::SRAM )
            .value("RRAM",Type::RRAM)
            .value("FeFET", Type::FeFET )
            .export_values();

    py::enum_<DeviceRoadmap>(m, "DeviceRoadmap")
            .value("HP", DeviceRoadmap::HP )
            .value("LSTP",DeviceRoadmap::LSTP)
            .export_values();

    py::enum_<DecoderMode>(m, "DecoderMode")
            .value("REGULAR_ROW", DecoderMode::REGULAR_ROW )
            .value("REGULAR_COL", DecoderMode::REGULAR_COL)
            .export_values();

    py::enum_<TransistorType>(m, "TransistorType")
            .value("conventional", TransistorType::conventional )
            .value("FET_2D",TransistorType::FET_2D)
            .value("TFET",TransistorType::TFET)
            .export_values();

    py::enum_<BusMode>(m, "BusMode")
            .value("HORIZONTAL", BusMode::HORIZONTAL)
            .value("VERTICAL", BusMode::VERTICAL);

    py::enum_<SpikingMode>(m, "SpikingMode")
            .value("NONSPIKING", SpikingMode::NONSPIKING)
            .value("SPIKING", SpikingMode::SPIKING);

    py::enum_<CellAccessType>(m, "CellAccessType")
            .value("CMOS_access", CellAccessType::CMOS_access)
            .value("BJT_access", CellAccessType::BJT_access)
            .value("diode_access", CellAccessType::diode_access)
            .value("none_access", CellAccessType::none_access);

    py::enum_<AreaModify>(m, "AreaModify")
            .value("NONE", AreaModify::NONE)
            .value("MAGIC",AreaModify::MAGIC)
            .value("OVERRIDE",AreaModify::OVERRIDE)
            .export_values();

    py::class_<InputParameter>(m, "InputParameter")
            .def(py::init<>())
            .def_readwrite("temperature", &InputParameter::temperature)
            .def_readwrite("transistorType", &InputParameter::transistorType)
            .def_readwrite("deviceRoadmap", &InputParameter::deviceRoadmap)
            .def_readwrite("processNode", &InputParameter::processNode);

    py::class_<Technology>(m, "Technology")
            //.def(py::init<int &, DeviceRoadmap &, TransistorType &>())
            .def(py::init<>())
            .def("Configure", &Technology::Initialize)
            .def_readwrite("featureSize", &Technology::featureSize);

    py::class_<Bus>(m, "Bus")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &Bus::Initialize)
            .def("CalculateArea", &Bus::CalculateArea,
                 py::arg("foldedratio"),
                 py::arg("overLap"))
            .def("CalculateLatency", &Bus::CalculateLatency)
            .def("CalculatePower", &Bus::CalculatePower)
            .def_readwrite("area", &Bus::area)
            .def_readwrite("width", &Bus::width)
            .def_readwrite("height", &Bus::height)
            .def_readwrite("numRow", &Bus::numRow)
            .def_readwrite("busWidth", &Bus::busWidth)
            .def_readwrite("readLatency", &Bus::readLatency)
            .def_readwrite("readDynamicEnergy", &Bus::readDynamicEnergy);

    py::class_<MemCell>(m, "MemCell")
            .def(py::init<>())
            .def_readwrite("cellBit", &MemCell::cellBit)
            .def_readwrite("widthAccessCMOS", &MemCell::widthAccessCMOS)
            .def_readwrite("widthSRAMCellNMOS", &MemCell::widthSRAMCellNMOS)
            .def_readwrite("widthSRAMCellPMOS", &MemCell::widthSRAMCellPMOS)
            .def_readwrite("memCellType", &MemCell::memCellType)
            .def_readwrite("widthInFeatureSize", &MemCell::widthInFeatureSize)
            .def_readwrite("heightInFeatureSize", &MemCell::heightInFeatureSize)
            .def_readwrite("minSenseVoltage", &MemCell::minSenseVoltage) 
            .def_readwrite("featureSize", &MemCell::featureSize)
            .def_readwrite("resistanceOn", &MemCell::resistanceOn)
            .def_readwrite("resistanceOff", &MemCell::resistanceOff)
            .def_readwrite("resistanceAvg", &MemCell::resistanceAvg)
            .def_readwrite("readVoltage", &MemCell::readVoltage)
            .def_readwrite("readPulseWidth", &MemCell::readPulseWidth)
            .def_readwrite("accessVoltage", &MemCell::accessVoltage)
            .def_readwrite("accessType", &MemCell::accessType)
            .def_readwrite("resistanceAccess", &MemCell::resistanceAccess)
            .def_readwrite("maxNumLevelLTP", &MemCell::maxNumLevelLTP)
            .def_readwrite("maxNumLevelLTD", &MemCell::maxNumLevelLTD)
            .def_readwrite("writeVoltage", &MemCell::writeVoltage)
            .def_readwrite("writePulseWidth", &MemCell::writePulseWidth)
            .def_readwrite("nonlinearIV", &MemCell::nonlinearIV)
            .def_readwrite("nonlinearity", &MemCell::nonlinearity);

    py::class_<SubArray>(m, "SubArray")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &SubArray::Initialize)
            .def("CalculateArea", &SubArray::CalculateArea)
            .def("CalculateLatency", &SubArray::CalculateLatency)
            .def("CalculatePower", &SubArray::CalculatePower)
            .def_readwrite("area", &SubArray::area)
            .def_readwrite("usedArea", &SubArray::usedArea)
            .def_readwrite("width", &SubArray::width)
            .def_readwrite("height", &SubArray::height)
            .def_readwrite("readLatency", &SubArray::readLatency)
            .def_readwrite("writeLatency", &SubArray::writeLatency)
            .def_readwrite("readDynamicEnergy", &SubArray::readDynamicEnergy)
            .def_readwrite("writeDynamicEnergy", &SubArray::writeDynamicEnergy)
            .def_readwrite("conventionalSequential", &SubArray::conventionalSequential)
            .def_readwrite("conventionalParallel", &SubArray::conventionalParallel)
            .def_readwrite("BNNsequentialMode", &SubArray::BNNsequentialMode)
            .def_readwrite("XNORsequentialMode", &SubArray::XNORsequentialMode)
            .def_readwrite("WeDummyCol", &SubArray::WeDummyCol)
            .def_readwrite("ADCmode", &SubArray::ADCmode)
            .def_readwrite("numRow", &SubArray::numRow)
            .def_readwrite("numCol", &SubArray::numCol)
            .def_readwrite("levelOutput", &SubArray::levelOutput)
            .def_readwrite("numColMuxed", &SubArray::numColMuxed)
            .def_readwrite("clkFreq", &SubArray::clkFreq)
            .def_readwrite("relaxArrayCellHeight", &SubArray::relaxArrayCellHeight)
            .def_readwrite("relaxArrayCellWidth", &SubArray::relaxArrayCellWidth)
            .def_readwrite("numReadPulse", &SubArray::numReadPulse)
            .def_readwrite("avgWeightBit", &SubArray::avgWeightBit)
            .def_readwrite("outputprecision", &SubArray::outputprecision)
            .def_readwrite("outputwidth", &SubArray::outputwidth)
            .def_readwrite("numCellPerSynapse", &SubArray::numCellPerSynapse)
            .def_readwrite("SARADC", &SubArray::SARADC)
            .def_readwrite("currentMode", &SubArray::currentMode)
            .def_readwrite("validated", &SubArray::validated)
            .def_readwrite("spikingMode", &SubArray::spikingMode)
            .def_readwrite("numReadCellPerOperationFPGA", &SubArray::numReadCellPerOperationFPGA)
            .def_readwrite("numWriteCellPerOperationFPGA", &SubArray::numWriteCellPerOperationFPGA)
            .def_readwrite("numReadCellPerOperationMemory", &SubArray::numReadCellPerOperationMemory)
            .def_readwrite("numWriteCellPerOperationMemory", &SubArray::numWriteCellPerOperationMemory)
            .def_readwrite("numReadCellPerOperationNeuro", &SubArray::numReadCellPerOperationNeuro)
            .def_readwrite("numWriteCellPerOperationNeuro", &SubArray::numWriteCellPerOperationNeuro)
            .def_readwrite("activityRowRead", &SubArray::activityRowRead)
            .def_readwrite("activityRowWrite", &SubArray::activityRowWrite)
            .def_readwrite("activityColWrite", &SubArray::activityColWrite)
            .def_readwrite("numWritePulse", &SubArray::numWritePulse)
            .def_readwrite("FPGA", &SubArray::FPGA)
            .def_readwrite("resCellAccess", &SubArray::resCellAccess)
            .def_readwrite("maxNumWritePulse", &SubArray::maxNumWritePulse)
            .def_readwrite("readLatencyADC", &SubArray::readLatencyADC)
            .def_readwrite("readDynamicEnergyADC", &SubArray::readDynamicEnergyADC)
            .def_readwrite("areaADC", &SubArray::areaADC);

    py::class_<Mux>(m, "Mux")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &Mux::Initialize,
                py::arg("_numInput"),
                py::arg("_numSelection"),
                py::arg("_resTg"),
                py::arg("_FPGA"))
            .def("CalculateArea", &Mux::CalculateArea,
                 py::arg("_newHeight"),
                 py::arg("_newWidth"),
                 py::arg("_option"))
            .def("CalculateLatency", &Mux::CalculateLatency,
                 py::arg("_rampInput"),
                 py::arg("_capLoad"),
                 py::arg("numRead"))
            .def("CalculatePower", &Mux::CalculatePower,
                 py::arg("numRead"))
            .def_readwrite("area", &Mux::area)
            .def_readwrite("width", &Mux::width)
            .def_readwrite("height", &Mux::height)
            .def_readwrite("readLatency", &Mux::readLatency);

    py::class_<RowDecoder>(m, "RowDecoder")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &RowDecoder::Initialize,
                py::arg("_mode"),
                py::arg("_numAddrRow"),
                py::arg("_MUX"),
                py::arg("_parallel"))
            .def("CalculateArea", &RowDecoder::CalculateArea,
                 py::arg("_newHeight"),
                 py::arg("_newWidth"),
                 py::arg("_option"))
            .def("CalculateLatency", &RowDecoder::CalculateLatency,
                 py::arg("_rampInput"),
                 py::arg("_capLoad1"),
                 py::arg("_capLoad2"),
                 py::arg("numRead"),
                 py::arg("numWrite"))
            .def("CalculatePower", &RowDecoder::CalculatePower,
                 py::arg("numRead"),
                 py::arg("numWrite"))
            .def_readwrite("area", &RowDecoder::area)
            .def_readwrite("width", &RowDecoder::width)
            .def_readwrite("height", &RowDecoder::height)
            .def_readwrite("readLatency", &RowDecoder::readLatency);

    py::class_<RecfgAdderTree>(m, "RecfgAdderTree")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &RecfgAdderTree::Initialize,
                py::arg("_numSubcoreRow"),
                py::arg("_numAdderBit"),
                py::arg("_numAdderTree"))
            .def("CalculateArea", &RecfgAdderTree::CalculateArea,
                 py::arg("_newHeight"),
                 py::arg("_newWidth"),
                 py::arg("_option"))
            .def("CalculateLatency", &RecfgAdderTree::CalculateLatency,
                 py::arg("numRead"),
                 py::arg("numUnitAdd"),
                 py::arg("numBitAdd"),
                 py::arg("_capLoad")
                 )
            .def("CalculatePower", &RecfgAdderTree::CalculatePower,
                 py::arg("numRead"),
                 py::arg("numUnitAdd"),
                 py::arg("numBitAdd"),
                 py::arg("numusedAdderTree") )
            .def_readwrite("area", &RecfgAdderTree::area)
            .def_readwrite("width", &RecfgAdderTree::width)
            .def_readwrite("height", &RecfgAdderTree::height)
            .def_readwrite("readLatency", &RowDecoder::readLatency);

    py::class_<Adder>(m, "Adder")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &Adder::Initialize)
            .def("CalculateArea", &Adder::CalculateArea,
                 py::arg("_newHeight"),
                 py::arg("_newWidth"),
                 py::arg("_option"))
            .def("CalculateLatency", &Adder::CalculateLatency,
                 py::arg("_rampInput"),
                 py::arg("_capLoad"),
                 py::arg("numRead"))
            .def("CalculatePower", &Adder::CalculatePower,
                 py::arg("numRead"),
                 py::arg("numAdderPerOperation"))
            .def_readwrite("area", &Adder::area)
            .def_readwrite("width", &Adder::width)
            .def_readwrite("height", &Adder::height)
            .def_readwrite("numAdder", &Adder::numAdder)
            .def_readwrite("readLatency", &Adder::readLatency)
            .def_readwrite("readDynamicEnergy", &Adder::readDynamicEnergy);

    py::class_<AdderTree>(m, "AdderTree")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &AdderTree::Initialize)
            .def("CalculateArea", &AdderTree::CalculateArea)
            .def("CalculateLatency", &AdderTree::CalculateLatency)
            .def("CalculatePower", &AdderTree::CalculatePower)
            .def_readwrite("area", &AdderTree::area)
            .def_readwrite("width", &AdderTree::width)
            .def_readwrite("height", &AdderTree::height)
            .def_readwrite("readLatency", &AdderTree::readLatency)
            .def_readwrite("numAdderTree", &AdderTree::numAdderTree)
            .def_readwrite("readDynamicEnergy", &AdderTree::readDynamicEnergy);

    py::class_<Multiplier>(m, "Multiplier")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &Multiplier::Initialize)
            .def("CalculateArea", &Multiplier::CalculateArea)
            .def("CalculateLatency", &Multiplier::CalculateLatency)
            .def("CalculatePower", &Multiplier::CalculatePower)
            .def_readwrite("area", &Multiplier::area)
            .def_readwrite("width", &Multiplier::width)
            .def_readwrite("height", &Multiplier::height)
            .def_readwrite("readLatency", &Multiplier::readLatency)
            .def_readwrite("readDynamicEnergy", &Multiplier::readDynamicEnergy)
            .def_readwrite("numMultiplier", &Multiplier::numMultiplier);

    py::class_<ShiftAdd>(m, "ShiftAdd")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &ShiftAdd::Initialize)
            .def("CalculateArea", &ShiftAdd::CalculateArea,
                 py::arg("_newHeight"),
                 py::arg("_newWidth"),
                 py::arg("_option"))
            .def("CalculateLatency", &ShiftAdd::CalculateLatency)
            .def("CalculatePower", &ShiftAdd::CalculatePower)
            .def_readwrite("area", &ShiftAdd::area)
            .def_readwrite("width", &ShiftAdd::width)
            .def_readwrite("height", &ShiftAdd::height)
            .def_readwrite("readLatency", &ShiftAdd::readLatency)
            .def_readwrite("readDynamicEnergy", &ShiftAdd::readDynamicEnergy);

    py::class_<DFF>(m, "DFF")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &DFF::Initialize)
            .def("CalculateArea", &DFF::CalculateArea,
                 py::arg("_newHeight"),
                 py::arg("_newWidth"),
                 py::arg("_option"))
            .def("CalculateLatency", &DFF::CalculateLatency)
            .def("CalculatePower", &DFF::CalculatePower)
            .def_readwrite("area", &DFF::area)
            .def_readwrite("width", &DFF::width)
            .def_readwrite("height", &DFF::height)
            .def_readwrite("readLatency", &DFF::readLatency)
            .def_readonly("numDff", &DFF::numDff)
            .def_readwrite("readDynamicEnergy", &DFF::readDynamicEnergy);
    py::class_<DRAM>(m, "DRAM")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &DRAM::Initialize)
            .def("CalculateLatency", &DRAM::CalculateLatency)
            .def("CalculatePower", &DRAM::CalculatePower)
            .def_readwrite("area", &DRAM::area)
            .def_readwrite("width", &DRAM::width)
            .def_readwrite("height", &DRAM::height)
            .def_readwrite("readLatency", &DRAM::readLatency)
            .def_readwrite("readDynamicEnergy", &DRAM::readDynamicEnergy);

    py::class_<Buffer>(m, "Buffer")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &Buffer::Initialize)
            .def("CalculateArea", &Buffer::CalculateArea)
            .def("CalculateLatency", &Buffer::CalculateLatency)
            .def("CalculatePower", &Buffer::CalculatePower)
            .def_readwrite("area", &Buffer::area)
            .def_readwrite("width", &Buffer::width)
            .def_readwrite("height", &Buffer::height)
            .def_readwrite("readLatency", &Buffer::readLatency)
            .def_readwrite("writeLatency", &Buffer::writeLatency)
            .def_readwrite("readDynamicEnergy", &Buffer::readDynamicEnergy)
            .def_readwrite("writeDynamicEnergy", &Buffer::writeDynamicEnergy)
            .def_readwrite("interface_width", &Buffer::interface_width);


    py::class_<HTree>(m, "HTree")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &HTree::Initialize)
            .def("CalculateArea", &HTree::CalculateArea)
            .def("CalculateLatency", &HTree::CalculateLatency)
            .def("CalculatePower", &HTree::CalculatePower)
            .def_readwrite("area", &HTree::area)
            .def_readwrite("width", &HTree::width)
            .def_readwrite("height", &HTree::height)
            .def_readwrite("readLatency", &HTree::readLatency)
            .def_readwrite("readDynamicEnergy", &HTree::readDynamicEnergy)
            .def_readwrite("busWidth", &HTree::busWidth);


    py::class_<BitShifter>(m, "BitShifter")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &BitShifter::Initialize)
            .def("CalculateArea", &BitShifter::CalculateArea)
            .def("CalculateLatency", &BitShifter::CalculateLatency)
            .def("CalculatePower", &BitShifter::CalculatePower)
            .def_readwrite("area", &BitShifter::area)
            .def_readwrite("width", &BitShifter::width)
            .def_readwrite("height", &BitShifter::height)
            .def_readwrite("numUnit", &BitShifter::numUnit)
            .def_readwrite("readLatency", &BitShifter::readLatency)
            .def_readwrite("readDynamicEnergy", &BitShifter::readDynamicEnergy);


    py::class_<MaxPooling>(m, "MaxPooling")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &MaxPooling::Initialize)
            .def("CalculateArea", &MaxPooling::CalculateArea)
            .def("CalculateUnitArea", &MaxPooling::CalculateUnitArea)
            .def("CalculateLatency", &MaxPooling::CalculateLatency)
            .def("CalculatePower", &MaxPooling::CalculatePower)
            .def_readwrite("area", &MaxPooling::area)
            .def_readwrite("width", &MaxPooling::width)
            .def_readwrite("height", &MaxPooling::height)
            .def_readwrite("readLatency", &MaxPooling::readLatency)
            .def_readwrite("readDynamicEnergy", &MaxPooling::readDynamicEnergy);

    py::class_<MultilevelSAEncoder>(m, "MultilevelSAEncoder")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &MultilevelSAEncoder::Initialize)
            .def("CalculateArea", &MultilevelSAEncoder::CalculateArea)
            .def("CalculateLatency", &MultilevelSAEncoder::CalculateLatency)
            .def("CalculatePower", &MultilevelSAEncoder::CalculatePower)
            .def_readwrite("area", &MultilevelSAEncoder::area)
            .def_readwrite("width", &MultilevelSAEncoder::width)
            .def_readwrite("height", &MultilevelSAEncoder::height)
            .def_readwrite("readLatency", &MultilevelSAEncoder::readLatency)
            .def_readwrite("readDynamicEnergy", &MultilevelSAEncoder::readDynamicEnergy);

    py::class_<MultilevelSenseAmp>(m, "MultilevelSenseAmp")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &MultilevelSenseAmp::Initialize)
            .def("CalculateArea", &MultilevelSenseAmp::CalculateArea)
            .def("CalculateLatency", &MultilevelSenseAmp::CalculateLatency)
            .def("CalculatePower", &MultilevelSenseAmp::CalculatePower)
            .def_readwrite("area", &MultilevelSenseAmp::area)
            .def_readwrite("width", &MultilevelSenseAmp::width)
            .def_readwrite("height", &MultilevelSenseAmp::height)
            .def_readwrite("readLatency", &MultilevelSenseAmp::readLatency)
            .def_readwrite("readDynamicEnergy", &MultilevelSenseAmp::readDynamicEnergy);
}