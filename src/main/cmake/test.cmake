## @author Ralf Quast
## @date 2021
## @copyright MIT License

function(add_unit_test NAME)
    add_executable(${NAME}
            ${ARGN}
            ${TEST}/fortran/test_mod.F90)
    add_test(NAME ${NAME} COMMAND ${NAME})
    set_tests_properties(${NAME} PROPERTIES LABELS unit TIMEOUT 60)
    add_custom_target(run${NAME} ${NAME}
            COMMENT "Running ${NAME}")
    add_dependencies(run${NAME} ${NAME})
endfunction()

add_custom_target(unittests ctest --output-on-failure --label-regex unit)

enable_testing()
