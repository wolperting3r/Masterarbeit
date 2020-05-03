module ml_mod
! Machine Learning 
implicit none

integer,allocatable,dimension(:)                :: ml_input
integer,allocatable,dimension(:)                :: ml_layer_nodes
character(len=4),allocatable,dimension(:)       :: ml_layer_activation
integer,allocatable,dimension(:)                :: ml_n_layerweights
integer,allocatable,dimension(:)                :: ml_n_biasweights
real*8,allocatable,dimension(:)                 :: ml_layerweights
real*8,allocatable,dimension(:)                 :: ml_biasweights
integer                                         :: ml_n_layers
integer                                         :: ml_n_features

! real*8, allocatable, dimension(:)               :: ml_input_layer
! real*8, allocatable, dimension(:)               :: ml_tmp
! real*8, allocatable, dimension(:)               :: ml_first_layer
! real*8, allocatable, dimension(:)               :: ml_second_layer
! real*8, allocatable, dimension(:)               :: ml_third_layer
! real*8, allocatable, dimension(:)               :: ml_output_layer 
integer                                         :: loop_counter
! integer                                         :: ml_out_bias_on

end module ml_mod
