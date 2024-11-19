!+ write feedback file for VISOP / SEVIRI reflectances
!
PROGRAM visop_iofdbk
!
! Description:
!   Write feedback file for VISOP / SEVIRI reflectances
!
!   2017.5 by L. Scheck (based on iofdbk.f90 by M. Sommer / P. Kostka (based on template by A. Rhodin))
!
!------------------------------------------------------------------------------
  use mo_fdbk,        only: setup_fdbk,       &! set up basic feedback-file data structure
                            create_fdbk,      &! set up feedback-file meta data
                            add_verification, &! add verification meta data to file
                            write_veri,       &! add verification data
                            close_fdbk,       &! close feedback file
                            cleanup_fdbk       ! deallocate components
  use mo_fdbk_io,     only: t_fdbk_data        ! type to hold feedback file content
  use mo_fdbk_rad,    only: setup_fdbk_rad ,  &! set up container for radiance data
                            write_fdbk_rad     ! write feedback file
  use mo_fdbk_tables, only: VN_RAWBT,         &! brightness temperature flag value
                            VN_REFL,          &! reflectivity
                            VE_DETERM,        &! deterministic run flag
                            ST_PASSIVE,       &! flag for passive data (monitoring only)
                            ST_REJECTED,      &! flag for rejected (bad) data
                            ST_DISMISS         ! flag for dismisssed data (dont write)
  implicit none

  !----------------------------------------------------------
  ! local variables passed to feedback file handling routines
  !----------------------------------------------------------
  integer            :: refdate         ! yyyymmdd start of forecast
  integer            :: reftime         ! hhmm     start of forecast
  integer            :: starttime       ! minutes  start of assimilation interval
  integer            :: endtime         ! minutes  end   of assimilation interval
  integer            :: n_fov           ! number of field of views
  integer            :: n_chan          ! number of channels
  character(len=80)  :: file            ! file name to write
  integer            :: ens_member      ! ensemble member
  integer            :: satid           ! WMO satellite id
  integer ,parameter :: m_chan = 3      ! max number of channels handled by this routine
  integer            :: varno (m_chan)  ! variable number
  integer            :: chan  (m_chan)  ! channel number
  integer            :: instr (m_chan)  ! instrument id
  real               :: e_o   (m_chan)  ! observation error
  integer            :: domain_size (3) ! domain size x,y,z
  real               :: resolution  (2) ! resolution dx, dy (degree)
  integer            :: id_veri         ! NetCDF varid for ensemble data
  integer            :: exp_id          ! experiment id
  integer            :: run_type        ! type of model run
  integer            :: run_class       ! class of model run
  integer            :: fc_time         ! hhmm forecast time at verification_ref_time

  real,    allocatable  :: fg_vec (:)      ! first guess, model equivalent
  real,    allocatable  :: obs_vec(:)      ! observations
  real,    allocatable  :: lat_vec(:)      ! obs. lat
  real,    allocatable  :: lon_vec(:)      ! obs. lon
  integer, allocatable  :: time_vec(:)     ! obs. time (rel. to ref. date/time)
  real,    allocatable  :: e_o_vec(:)      ! obs. error
  real,    allocatable  :: h_loc_vec(:)    ! hor. localization [km]
  real,    allocatable  :: v_loc_vec(:)    ! vert. localization [ln p]

  character(len=8)   :: prefix          ! prefix for fof-files

  type(t_fdbk_data) :: fdbk_data        ! derived type to store file content
  character(len=12) :: datetime         ! date&time string
  integer           :: n_data           ! total # of data (fofs * channels)

  !----------------------
  ! additional quantities
  !----------------------
  integer :: ierr, Iges, kenkf, i, j
  integer :: ix, iy, stridex, stridey  ! reflectance array dimensions, strides for thinning

  real :: lonmin, lonmax, latmin, latmax ! observations outside of this lat-lon-rectangle are ignored

  integer, parameter :: ix_max=5000
  integer :: asswinmin, fctimehhmm     ! assimilation window [min], fcst time [hhmm] relative to refdate/time
  real    :: obserr                    ! observation error
  integer :: nnml                      ! default namelist unit
  integer :: iostat                    ! error return code
  character(len=80), parameter :: nmlfile='namelist_visop_iofdbk'         ! name of file to open
  real :: tmp(ix_max)

  integer :: iobs ! number of observations (after thinning, spatial restriction) (<= ix * iy)


  !--------------
  ! read namelist
  !--------------

  NAMELIST/VISOPIOFDBK/Iges, kenkf, exp_id, refdate, reftime, n_data, asswinmin, fctimehhmm
  nnml = 99
  open (nnml, file=trim(nmlfile), status='old', action='read', iostat=ierr)
  if (ierr/=0) stop 'cannot open namelist file '//trim(nmlfile)
  read (nnml, nml=VISOPIOFDBK, iostat=ierr)
  if (ierr>0) stop 'ERROR in namelist /VISOPIOFDBK/'

  print *, '============================ VISOP_IOFDBK ==============================='
  print *, '==  processing ensemble member', Iges, ' out of ', kenkf
  print *, '==  reference date and time          : ', refdate, reftime
  print *, '==  number of observations           : ', n_data
  print *, '==  assimilation window length [min] : ', asswinmin
  print *, '==  forecast time [hhmm]             : ', fctimehhmm
  print *, '==  experiment id                    : ', exp_id


  !------------------------------------------------------------------
  ! read observations and first guess reflectances generated by visop
  !------------------------------------------------------------------

  allocate( time_vec( n_data) )
  allocate( obs_vec(  n_data) )
  allocate( fg_vec(   n_data) )
  allocate( e_o_vec(  n_data) )
  allocate( h_loc_vec(n_data) )
  allocate( v_loc_vec(n_data) )
  allocate( lat_vec(  n_data) )
  allocate( lon_vec(  n_data) )

  call read_visop_output( time_vec, obs_vec, fg_vec, e_o_vec, h_loc_vec, v_loc_vec, lat_vec, lon_vec, n_data ) 
  print *, '==  Minimum in fg    : ',  minval( fg_vec(:)),   '. Maximum in fg    : ', maxval( fg_vec(:))
  print *, '==  Minimum in obs   : ',  minval(obs_vec(:)),   '. Maximum in obs   : ', maxval(obs_vec(:))
  print *, '==  Minimum in e_o   : ',  minval(e_o_vec(:)),   '. Maximum in e_o   : ', maxval(e_o_vec(:))
  print *, '==  Minimum in h_loc : ',  minval(h_loc_vec(:)), '. Maximum in h_loc : ', maxval(h_loc_vec(:))
  print *, '==  Minimum in v_loc : ',  minval(v_loc_vec(:)), '. Maximum in v_loc : ', maxval(v_loc_vec(:))
  print *, '==  Minimum in lat   : ',  minval(lat_vec(:)),   '. Maximum in lat   : ', maxval(lat_vec(:))
  print *, '==  Minimum in lon   : ',  minval(lon_vec(:)),   '. Maximum in lon   : ', maxval(lon_vec(:))

  !--------------------------------------------------------------------
  ! Set example values.
  ! In a real application these values have to be derived from the data
  !   date & time of analysis etc.
  !--------------------------------------------------------------------
  prefix      = 'vis_'                         ! arbitrary file prefix
  ens_member  =  Iges                          ! ensemble member to write
  if( Iges == 0 ) ens_member = VE_DETERM
! ens_member  = VE_DETERM                      ! use this value for deterministic run deterministic run
!   refdate     = 20090807                     ! reference date, e.g. start of assimilation window
!   reftime     = 1500                         ! reference time, e.g. start of assimilation window
  starttime   = 1                              ! start of assimilation window (minutes) relative to refdate/ttime
  endtime     = asswinmin                      ! end   of assimilation window (minutes)
  n_fov       = n_data                           ! number of field of views
  n_chan      = 1                              ! number of channels
!  satid       = 72 ! = MSG2 (was 70)           ! WMO satellite id
  satid       = 57 ! = MSG3                     ! WMO satellite id
  varno (1)   = VN_REFL                       ! observed quantity (VN_REFL, VN_RAWBT, VN_RADIANCE)
  chan  (1)   = 1 ! 1=3.9mu (0.6mu not yet available)   ! channel number
  instr (1)   = 207                              ! WMO instrument number
  e_o   (1)   = obserr                         ! observation error
  domain_size = (/421, 461, 50/)               ! domain size (COSMO-DE hardcoded)
  resolution  = (/0.025, 0.025/)               ! resolution  (COSMO-DE hardcoded)
  run_type    = 1                              ! take from COSMO model
  run_class   = 1                              ! take from COSMO model
  fc_time     = fctimehhmm ! LS: was 300       ! forecast time at ref_time hhmm     LS:???


  !-----------------
  ! set up meta data
  !-----------------

  ! set file name
  write (datetime,'(i8.8,i4.4)') refdate, reftime
  if (ens_member > 0) then
    write (file,'(a,a,a,i3.3,a)') trim(prefix), datetime, '00_ens', ens_member, '.nc'
  else
    write (file,'(a,a,a,i3.3,a)') trim(prefix), datetime, '00.nc'
  endif
  print *, '==  writing file ', trim(file)

  ! intialize NetCDF file structure
  call setup_fdbk   (fdbk_data% f% nc)

  call create_fdbk  (fdbk_data% f,            &! feedback file meta data
                     trim(file),              &! path
                     'VIS_NR',                &! model
                     '00.00',                 &! model version
                     'LMU Munich',            &! institution
                     n_fov,                   &! d_hdr
                     n_data,                  &! d_body
                     refdate,                 &! reference date
                     reftime,                 &! reference time
                     starttime,               &! start of verification
                     endtime,                 &! end   of verification
                     resolution,              &! resolution
                     domain_size,             &! domain size
                     'KENDA first guess',     &! comment (for history)
                     datetime,                &! time    (for history)
              pole = (/40., -170./),          &! location of pole    (hardcoded)
        lower_left = (/-5., -5./),            &! lower left  (lat,lon hardcoded)
       upper_right = (/6.5, 5.5/),            &! upper right (lat,lon hardcoded)
            create = .false.                  )! postpone actual file creation

  !-----------------------------------
  ! set up container for radiance data
  !-----------------------------------
  call setup_fdbk_rad (fdbk_data, n_fov, n_chan, refdate, reftime,  &
                       satid, varno, chan, instr, e_o)


  !----------------------------------------------
  ! fill in header data.
  ! coordinates are mandatory.
  ! other data is currently not required by LETKF
  !   (but may be used later)
  !----------------------------------------------


  ! LS: fill with something
  fdbk_data% h% sun_zenit     = 30.0          ! solar zenith angle
  fdbk_data% h% mdlsfc        = 1          ! model surface flag
  fdbk_data% h% z_modsurf     = 500          ! model surface height
  fdbk_data% h% index_x       = 1          ! model grid index
  fdbk_data% h% index_y       = 1          ! model grid index
  fdbk_data% h% surftype      = 4          ! surface type from data
  fdbk_data% h% sat_zenit     = 45.0          ! satellite zenith angle

  fdbk_data% h% lat  = lat_vec(:)
  fdbk_data% h% lon  = lon_vec(:)
  fdbk_data% h% time = time_vec(:)

  fdbk_data% h% ident = satid
  fdbk_data% h% instype = instr(1)


  !------------------------------------------------------------------------
  ! fill in body data (observed values and nominal height for localisation)
  ! in this example all the same for same channel
  !------------------------------------------------------------------------

  fdbk_data% b (1::n_chan) % obs = obs_vec(:)
  fdbk_data% b (1::n_chan) % e_o = e_o_vec(:)
  fdbk_data% b (1::n_chan) % plevel = 50000.   ! nominal height (Pa)

  fdbk_data% b (1::n_chan) % h_loc = h_loc_vec(:)
  fdbk_data% b (1::n_chan) % v_loc = v_loc_vec(:)

  !-------------------------------------------------------------------
  ! set some quality flags for all channels of the fov:
  !   may be set fron fg check, provider quality flags etc.
  !   DISMISS:  data will not be written to file, not yet implemented.
  !   PASSIVE:  data will not be assimilated, just monitored.
  !   REJECTED: data is of bad quality, not assimilated, monitored.
  !-------------------------------------------------------------------
! fdbk_data% h (2) % r_state = ST_DISMISS  !+++ dismiss doesnt work yet ++++
!  fdbk_data% h (4) % r_state = ST_PASSIVE

  !---------------------------------------------------
  ! set some quality flags for individual fof/channels
  !---------------------------------------------------
!  fdbk_data% b (1) % state = ST_PASSIVE    ! 1st channel 1st fof
!  fdbk_data% b (6) % state = ST_REJECTED   ! 2nd channel 3rd fof

  !------------------------
  ! actually write the file
  !------------------------

  call write_fdbk_rad (fdbk_data)

  !-------------------------
  ! fill in model equivalent
  !-------------------------
  call add_verification (fdbk_data% f,          &!
                         'COSMO',               &! model
                         run_type,              &!
                         run_class,             &!
                         datetime,              &! initial_date
                         fc_time,               &! fc_time
                         resolution,            &! dx, dy (degree)
                         domain_size,           &! nx,ny,nz
                         'KENDA radiance data', &! description
                         ens_member,            &! ensemble id
                         exp_id,                &! experiment id
                         id_veri                )! NetCDF varid

  call write_veri   (fdbk_data% f, 1, fg_vec(:))

  call close_fdbk   (fdbk_data% f)
  call cleanup_fdbk (fdbk_data% f)

  print *, '==  done.'

contains

  subroutine read_visop_output( time_vec, obs_vec, fg_vec, e_o_vec, h_loc_vec, v_loc_vec, lat_vec, lon_vec, n_data ) 

    integer, intent(inout)  :: time_vec( n_data )
    real,    intent(inout)  :: obs_vec( n_data)
    real,    intent(inout)  :: fg_vec(  n_data)
    real,    intent(inout)  :: e_o_vec( n_data)
    real,    intent(inout)  :: h_loc_vec( n_data)
    real,    intent(inout)  :: v_loc_vec( n_data)
    real,    intent(inout)  :: lat_vec( n_data)
    real,    intent(inout)  :: lon_vec( n_data)
    integer, intent(in)     :: n_data
    integer                 :: n_data_file
    integer, parameter      :: unit=98 
    
    open(unit, file='obsmeq.dat', status='old', action='read')
    
    read(unit,*) n_data_file
    if( n_data_file .ne. n_data ) then
       stop 'n_data in file does not match n_data in namelist'
    end if
    
    do i = 1, n_data
       read(unit,*) time_vec(i), obs_vec(i), fg_vec(i), e_o_vec(i), h_loc_vec(i), v_loc_vec(i), lat_vec(i), lon_vec(i)
       !print *, i, time_vec(i), obs_vec(i), fg_vec(i), e_o_vec(i), h_loc_vec(i), v_loc_vec(i), lat_vec(i), lon_vec(i)
    end do
    
    close(unit)

  end subroutine read_visop_output

end program visop_iofdbk
