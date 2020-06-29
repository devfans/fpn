// Populate from a = (PI/4) * 0 / 64, tan(a) to a = (PI/4) * 64 / 64, tan(a)
const populate = () => {
  const angles = [];
  const tans = [];
	const coss = [];
  let frac = 2 ** 12;
  for (let i = 0; i < 65; i++) {
    const a = i * Math.PI / (4 * 64);
    const t = Math.tan(a);
    const c = Math.cos(a);
    angles.push(Math.round(frac * a))
    tans.push(Math.round(frac * t))
    coss.push(Math.round(frac * c))
  }
  console.log("/// Constants used to convert values from cartesian coordinates to polar coordinates");
  console.log("/// Constant tan(a) and cos(a) values with a from (PI/4) * 0 / 64 to (PI/4) * 64 / 64")
  /*
  console.log("const POLAR_ANGLES: [i64; 65] = [")
  angles.forEach(a => console.log(`    ${a}i64,`))
  console.log("];")
  */
  console.log("const POLAR_TANS: [i64; 65] = [")
  tans.forEach(a => console.log(`    ${a}i64,`))
  console.log("];")
	console.log("const POLAR_COSS: [i64; 65] = [")
  coss.forEach(a => console.log(`    ${a}i64,`))
  console.log("];")
  frac = 2 ** 56;
  const pi = Math.round(frac * Math.PI);
  const pi2 = Math.round(frac * (Math.PI * Math.PI));
  const pix2 = Math.round(2 * frac * Math.PI);
  const pihalf = Math.round(0.5 * frac * Math.PI);
  const piquad = Math.round(0.25 * frac * Math.PI);
  console.log("/// PI << 56, (PI ** 2) << 56, (PI * 2) << 56, etc");
  console.log(`const PIS56: i64 = ${pi}i64;`);
  console.log(`const PIS56_SQUARE: i64 = ${pi2}i64;`);
  console.log(`const PIS56_DOUBLE: i64 = ${pix2}i64;`);
  console.log(`const PIS56_HALF: i64 = ${pihalf}i64;`);
  console.log(`const PIS56_QUAD: i64 = ${piquad}i64;`);
}

populate();
