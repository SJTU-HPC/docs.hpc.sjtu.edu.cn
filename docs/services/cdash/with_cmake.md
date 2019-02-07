# CTest/CDash with CMake

If your project already uses CMake ([documentation](https://cmake.org/cmake/help/latest/)
and [wiki](https://gitlab.kitware.com/cmake/community/wikis/home))
to generate a build system, using the CDash dashboard at NERSC is greatly simplified.
For general instructions on creating CMake tests, see the [Testing section](https://cmake.org/cmake-tutorial/#s3) of the CMake Tutorial. Be sure to add:

```cmake
enable_testing()
include(CTest)
```

to the the top-level CMakeLists.txt before adding any tests. This will generate a `DartConfiguration.tcl` file needed for dashboard submission.

When submitting to a dashboard, provide a `CTestConfig.cmake` that provides the information about
where the dashboard is located and the submission configuration. For the [NERSC dashboard](https://cdash.nersc.gov), customize `CTEST_PROJECT_NAME` below after creating a project, place the file in the top-level directory of the source code, and copy this file into the build directory (if different than the source directory).

## Example CTestConfig.cmake

```cmake
#===================================================================================
# Replace the value for CTEST_PROJECT_NAME with your project name at cdash.nersc.gov
#===================================================================================

set(CTEST_PROJECT_NAME          "TiMemory")
set(CTEST_NIGHTLY_START_TIME    "01:00:00 UTC")
set(CTEST_DROP_METHOD           "https")
set(CTEST_DROP_SITE             "cdash.nersc.gov")
set(CTEST_DROP_LOCATION         "/submit.php?project=${CTEST_PROJECT_NAME}")
set(CTEST_DROP_SITE_CDASH       TRUE)
set(CTEST_CDASH_VERSION         "1.6")
set(CTEST_CDASH_QUERY_VERSION   TRUE)
```

Add a `CTestCustom.cmake` file for customizaton. This file can define some settings CTest settings or define variables needed by CTest/CDash.

## Example CTestCustom.cmake

```cmake
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS           "200" )
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS         "500" )
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE    "104857600") # 100 MB
set(CTEST_CUSTOM_COVERAGE_EXCLUDE                   "")

# either customize these directly or write as CMake Template
# and use configure_file(... @ONLY) with CMake
set(CTEST_SOURCE_DIRECTORY   "/path/to/source")
set(CTEST_BINARY_DIRECTORY   "/path/to/source")
# build options
set(OPTION_BUILD             "-j8")
# define generator (optional), e.g. default to 'Unix Makefiles' on UNIX, Visual Studio on Windows
set(CTEST_GENERATOR          "...")
# submit under Continuous, Nightly (default), Experimental
set(CTEST_MODEL              "Continuous")
# define how to checkout code, e.g. copy a directory, git pull, svn co, etc.
set(CTEST_CHECKOUT_COMMAND   "...")
# define how to update (optional), e.g. git checkout <git-branch>
set(CTEST_UPDATE_COMMAND     "...")
# define how to configure (e.g. cmake -DCMAKE_INSTALL_PREFIX=...)
set(CTEST_CONFIGURE_COMMAND  "...")
# the name of the build
set(CTEST_BUILD_NAME         "...")
# how to configure
set(CTEST_CONFIGURE_COMMAND  "...")
# how to build
set(CTEST_BUILD_COMMAND      "...")
# default max time each tests can run (in seconds)
set(CTEST_TIMEOUT            "7200")
# locale to English
set(ENV{LC_MESSAGES}         "en_EN")
```

For more information on the commands to set, the [wiki for using CTest without CMake](https://gitlab.kitware.com/cmake/community/wikis/doc/ctest/Using-CTEST-and-CDASH-without-CMAKE)
can be useful.

## Example Dashboard Submission Script
Write a dashboard submission script. In the example below, it is assumed `CTestConfig.cmake` and `CTestCustom.cmake` exist in `CTEST_BINARY_DIRECTORY` (processes in `ctest_read_custom_files` step).
The `RETURN_VALUE` is not necessary but can be used to stop further processing if something failed,
e.g. don't run testing if build failed.

```cmake
ctest_read_custom_files(${CTEST_BINARY_DIRECTORY})

ctest_start             (${CTEST_MODEL} TRACK ${CTEST_MODEL})
ctest_configure         (BUILD ${CTEST_BINARY_DIRECTORY} RETURN_VALUE ret_con)
ctest_build             (BUILD ${CTEST_BINARY_DIRECTORY} RETURN_VALUE ret_bld)

if(ret_bld)
    # add as desired
    ctest_test              (BUILD ${CTEST_BINARY_DIRECTORY} RETURN_VALUE ret_tst)
    ctest_memcheck          (BUILD ${CTEST_BINARY_DIRECTORY} RETURN_VALUE ret_mem)
    ctest_coverage          (BUILD ${CTEST_BINARY_DIRECTORY} RETURN_VALUE ret_cov)

    # attach build notes if desired, e.g. performance info, output files from tests
    list(APPEND CTEST_NOTES_FILES "/file/to/attach/as/build-note")
endif()

# standard submit
ctest_submit(RETURN_VALUE ret_sub)
# if dashboard requires a token that restricts who can submit to dashboard
ctest_submit(RETURN_VALUE ret_sub HTTPHEADER "Authorization: Bearer ${CTEST_TOKEN}")

```

## Testing + Submission

If the dashboard submission script is named `Dashboard.cmake`. One would run:

`ctest -S Dashboard.cmake`

and CTest will checkout the code into `CTEST_SOURCE_DIRECTORY`, configure, build, test, etc. in
`CTEST_BINARY_DIRECTORY` and submit the results to the dashboard. More information and additional
options can always be found in `ctest` manual pages.
