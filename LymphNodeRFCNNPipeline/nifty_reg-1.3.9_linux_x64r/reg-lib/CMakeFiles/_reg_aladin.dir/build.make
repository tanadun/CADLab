# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r

# Include any dependencies generated for this target.
include reg-lib/CMakeFiles/_reg_aladin.dir/depend.make

# Include the progress variables for this target.
include reg-lib/CMakeFiles/_reg_aladin.dir/progress.make

# Include the compile flags for this target's objects.
include reg-lib/CMakeFiles/_reg_aladin.dir/flags.make

reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o: reg-lib/CMakeFiles/_reg_aladin.dir/flags.make
reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o: /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg/reg-lib/_reg_aladin.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o"
	cd /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r/reg-lib && /usr/bin/g++-4.4   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o -c /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg/reg-lib/_reg_aladin.cpp

reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.i"
	cd /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r/reg-lib && /usr/bin/g++-4.4  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg/reg-lib/_reg_aladin.cpp > CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.i

reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.s"
	cd /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r/reg-lib && /usr/bin/g++-4.4  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg/reg-lib/_reg_aladin.cpp -o CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.s

reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o.requires:
.PHONY : reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o.requires

reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o.provides: reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o.requires
	$(MAKE) -f reg-lib/CMakeFiles/_reg_aladin.dir/build.make reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o.provides.build
.PHONY : reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o.provides

reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o.provides.build: reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o

reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o: reg-lib/CMakeFiles/_reg_aladin.dir/flags.make
reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o: /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg/reg-lib/_reg_aladin_sym.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o"
	cd /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r/reg-lib && /usr/bin/g++-4.4   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o -c /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg/reg-lib/_reg_aladin_sym.cpp

reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.i"
	cd /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r/reg-lib && /usr/bin/g++-4.4  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg/reg-lib/_reg_aladin_sym.cpp > CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.i

reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.s"
	cd /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r/reg-lib && /usr/bin/g++-4.4  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg/reg-lib/_reg_aladin_sym.cpp -o CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.s

reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o.requires:
.PHONY : reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o.requires

reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o.provides: reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o.requires
	$(MAKE) -f reg-lib/CMakeFiles/_reg_aladin.dir/build.make reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o.provides.build
.PHONY : reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o.provides

reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o.provides.build: reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o

# Object files for target _reg_aladin
_reg_aladin_OBJECTS = \
"CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o" \
"CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o"

# External object files for target _reg_aladin
_reg_aladin_EXTERNAL_OBJECTS =

reg-lib/lib_reg_aladin.so: reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o
reg-lib/lib_reg_aladin.so: reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o
reg-lib/lib_reg_aladin.so: reg-lib/CMakeFiles/_reg_aladin.dir/build.make
reg-lib/lib_reg_aladin.so: reg-lib/lib_reg_localTransformation.a
reg-lib/lib_reg_aladin.so: reg-lib/lib_reg_blockMatching.a
reg-lib/lib_reg_aladin.so: reg-lib/lib_reg_resampling.a
reg-lib/lib_reg_aladin.so: reg-lib/lib_reg_globalTransformation.a
reg-lib/lib_reg_aladin.so: reg-lib/lib_reg_ssd.a
reg-lib/lib_reg_aladin.so: reg-lib/lib_reg_mutualinformation.a
reg-lib/lib_reg_aladin.so: reg-lib/lib_reg_tools.a
reg-lib/lib_reg_aladin.so: reg-io/lib_reg_ReadWriteImage.a
reg-lib/lib_reg_aladin.so: reg-io/png/libreg_png.a
reg-lib/lib_reg_aladin.so: /usr/lib/x86_64-linux-gnu/libpng.so
reg-lib/lib_reg_aladin.so: reg-io/nrrd/libreg_nrrd.a
reg-lib/lib_reg_aladin.so: reg-lib/lib_reg_tools.a
reg-lib/lib_reg_aladin.so: reg-lib/lib_reg_maths.a
reg-lib/lib_reg_aladin.so: reg-io/nifti/libreg_nifti.a
reg-lib/lib_reg_aladin.so: reg-io/nrrd/libreg_NrrdIO.a
reg-lib/lib_reg_aladin.so: reg-lib/CMakeFiles/_reg_aladin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library lib_reg_aladin.so"
	cd /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r/reg-lib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/_reg_aladin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
reg-lib/CMakeFiles/_reg_aladin.dir/build: reg-lib/lib_reg_aladin.so
.PHONY : reg-lib/CMakeFiles/_reg_aladin.dir/build

reg-lib/CMakeFiles/_reg_aladin.dir/requires: reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin.cpp.o.requires
reg-lib/CMakeFiles/_reg_aladin.dir/requires: reg-lib/CMakeFiles/_reg_aladin.dir/_reg_aladin_sym.cpp.o.requires
.PHONY : reg-lib/CMakeFiles/_reg_aladin.dir/requires

reg-lib/CMakeFiles/_reg_aladin.dir/clean:
	cd /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r/reg-lib && $(CMAKE_COMMAND) -P CMakeFiles/_reg_aladin.dir/cmake_clean.cmake
.PHONY : reg-lib/CMakeFiles/_reg_aladin.dir/clean

reg-lib/CMakeFiles/_reg_aladin.dir/depend:
	cd /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9/nifty_reg/reg-lib /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r/reg-lib /home/rothhr/Code/CADLab-GitHub/CADLab-git/LymphNodeRFCNNPipeline/nifty_reg-1.3.9_linux_x64r/reg-lib/CMakeFiles/_reg_aladin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : reg-lib/CMakeFiles/_reg_aladin.dir/depend
