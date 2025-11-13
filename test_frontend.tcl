#!/usr/bin/env tclsh
package require Tk

# --- GUI Setup ---
wm title . "Polynomial Fitter Test Suite"
set PADDING 10

# --- Fit Controls ---
labelframe .fit -text "Fit Polynomial" -padx $PADDING -pady $PADDING
grid .fit -row 0 -column 0 -padx $PADDING -pady $PADDING -sticky "ew"

label .fit.x_label -text "X Data File:"
entry .fit.x_entry -width 40
label .fit.y_label -text "Y Data File:"
entry .fit.y_entry -width 40
label .fit.degree_label -text "Degree:"
entry .fit.degree_entry -width 5
button .fit.run -text "Run Fit" -command run_fit

grid .fit.x_label -row 0 -column 0 -sticky "w"
grid .fit.x_entry -row 0 -column 1 -sticky "ew"
grid .fit.y_label -row 1 -column 0 -sticky "w"
grid .fit.y_entry -row 1 -column 1 -sticky "ew"
grid .fit.degree_label -row 2 -column 0 -sticky "w"
grid .fit.degree_entry -row 2 -column 1 -sticky "w"
grid .fit.run -row 3 -column 1 -pady $PADDING -sticky "e"

# --- Compose Controls ---
labelframe .compose -text "Compose Polynomials" -padx $PADDING -pady $PADDING
grid .compose -row 1 -column 0 -padx $PADDING -pady $PADDING -sticky "ew"

label .compose.p1_coeffs_label -text "P1 Coeffs File:"
entry .compose.p1_coeffs_entry -width 40
label .compose.p1_delta_label -text "P1 Delta:"
entry .compose.p1_delta_entry -width 10
label .compose.p2_coeffs_label -text "P2 Coeffs File:"
entry .compose.p2_coeffs_entry -width 40
label .compose.p2_delta_label -text "P2 Delta:"
entry .compose.p2_delta_entry -width 10
label .compose.degree_label -text "Degree:"
entry .compose.degree_entry -width 5
button .compose.run -text "Run Compose" -command run_compose

grid .compose.p1_coeffs_label -row 0 -column 0 -sticky "w"
grid .compose.p1_coeffs_entry -row 0 -column 1 -sticky "ew"
grid .compose.p1_delta_label -row 1 -column 0 -sticky "w"
grid .compose.p1_delta_entry -row 1 -column 1 -sticky "w"
grid .compose.p2_coeffs_label -row 2 -column 0 -sticky "w"
grid .compose.p2_coeffs_entry -row 2 -column 1 -sticky "ew"
grid .compose.p2_delta_label -row 3 -column 0 -sticky "w"
grid .compose.p2_delta_entry -row 3 -column 1 -sticky "w"
grid .compose.degree_label -row 4 -column 0 -sticky "w"
grid .compose.degree_entry -row 4 -column 1 -sticky "w"
grid .compose.run -row 5 -column 1 -pady $PADDING -sticky "e"

# --- Output ---
labelframe .output -text "Output" -padx $PADDING -pady $PADDING
grid .output -row 2 -column 0 -padx $PADDING -pady $PADDING -sticky "nsew"
text .output.text -width 80 -height 10 -wrap word
grid .output.text -row 0 -column 0 -sticky "nsew"
grid columnconfigure . 0 -weight 1
grid rowconfigure .output 0 -weight 1

# --- Procedures ---
proc run_fit {} {
    set x_file [.fit.x_entry get]
    set y_file [.fit.y_entry get]
    set degree [.fit.degree_entry get]

    if {$x_file == "" || $y_file == "" || $degree == ""} {
        .output.text insert end "Error: All fit fields are required.\n"
        return
    }

    set cmd "./run_test fit $x_file $y_file $degree"
    .output.text insert end "Running: $cmd\n"
    set output [exec {*}$cmd]
    .output.text insert end "Output:\n$output\n"

    set coeffs [split [string trim $output] "\n"]
    set poly "p(x) = "
    set i 0
    foreach c $coeffs {
        if {$i == 0} {
            set poly "$poly $c"
        } else {
            set poly "$poly + $c * x**$i"
        }
        incr i
    }

    set gnuplot_script "
        set title 'Polynomial Fit'
        set xlabel 'x'
        set ylabel 'y'
        $poly
        plot '$x_file' using 1:2 with points title 'Data', p(x) with lines title 'Fit'
    "
    set gnuplot_file [open "| gnuplot -persist" "w"]
    puts $gnuplot_file $gnuplot_script
    close $gnuplot_file
}

proc run_compose {} {
    set p1_coeffs_file [.compose.p1_coeffs_entry get]
    set p1_delta [.compose.p1_delta_entry get]
    set p2_coeffs_file [.compose.p2_coeffs_entry get]
    set p2_delta [.compose.p2_delta_entry get]
    set degree [.compose.degree_entry get]

    if {$p1_coeffs_file == "" || $p1_delta == "" || $p2_coeffs_file == "" || $p2_delta == "" || $degree == ""} {
        .output.text insert end "Error: All compose fields are required.\n"
        return
    }

    set cmd "./run_test compose $p1_coeffs_file $p1_delta $p2_coeffs_file $p2_delta $degree"
    .output.text insert end "Running: $cmd\n"
    set output [exec {*}$cmd]
    .output.text insert end "Output:\n$output\n"

    set coeffs [split [string trim $output] "\n"]
    set p1_coeffs [split [string trim [exec cat $p1_coeffs_file]] "\n"]
    set p2_coeffs [split [string trim [exec cat $p2_coeffs_file]] "\n"]

    set p1 "p1(x) = "
    set i 0
    foreach c $p1_coeffs {
        if {$i == 0} {
            set p1 "$p1 $c"
        } else {
            set p1 "$p1 + $c * x**$i"
        }
        incr i
    }

    set p2 "p2(x) = "
    set i 0
    foreach c $p2_coeffs {
        if {$i == 0} {
            set p2 "$p2 $c"
        } else {
            set p2 "$p2 + $c * x**$i"
        }
        incr i
    }

    set p "p(x) = "
    set i 0
    foreach c $coeffs {
        if {$i == 0} {
            set p "$p $c"
        } else {
            set p "$p + $c * x**$i"
        }
        incr i
    }

    set gnuplot_script "
        set title 'Composed Polynomials'
        set xlabel 'x'
        set ylabel 'y'
        set xrange [0:1]
        $p1
        $p2
        $p
        plot p1(x) with lines title 'P1', p2(x) with lines title 'P2', p(x) with lines dashtype 2 title 'Composed'
    "
    set gnuplot_file [open "| gnuplot -persist" "w"]
    puts $gnuplot_file $gnuplot_script
    close $gnuplot_file
}
