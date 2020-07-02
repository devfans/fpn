// Populate from a = (PI/4) * 0 / 64, tan(a) to a = (PI/4) * 64 / 64, tan(a)
const populate = () => {
  const angles = [];
  const tans = [];
	const coss = [];
	const sins = [];
  let frac = 2 ** 12;
  for (let i = 0; i < 65; i++) {
    const a = i * Math.PI / (4 * 64);
    const t = Math.tan(a);
    const c = Math.cos(a);
    const s = Math.sin(a);
    angles.push(Math.round(frac * a))
    tans.push(Math.round(frac * t))
    coss.push(Math.round(frac * c))
    sins.push(Math.round(frac * s))
  }
  for (let i = 65; i < 128; i++) {
    const a = i * Math.PI / (4 * 64);
    const t = Math.tan(a);
    tans.push(Math.round(frac * t))
  }
  console.log("/// Constants used to convert values from cartesian coordinates to polar coordinates");
  console.log("/// Constant tan(a) values with a from (PI/4) * 0 / 64 to (PI/4) * 128 / 64")
  console.log("/// Constant sin(a) and cos(a) values with a from (PI/4) * 0 / 64 to (PI/4) * 64 / 64")
  /*
  console.log("const POLAR_ANGLES: [i64; 65] = [")
  angles.forEach(a => console.log(`    ${a}i64,`))
  console.log("];")
  */
  console.log("pub const TANS: [i64; 129] = [")
  for (let i = 0; i < tans.length ; i+=10)
    console.log('    ' + tans.slice(i, i + 10).join('i64, ') + 'i64,')
  console.log("    std::i64::MAX,");
  console.log("];")
	console.log("pub const COSS: [i64; 65] = [")
  for (let i = 0; i < coss.length ; i+=10)
    console.log('    ' + coss.slice(i, i + 10).join('i64, ') + 'i64,')
  console.log("];")
  console.log("pub const SINS: [i64; 65] = [")
  for (let i = 0; i < sins.length ; i+=10)
    console.log('    ' + sins.slice(i, i + 10).join('i64, ') + 'i64,')
  console.log("];")

  frac = 2 ** 56;
  const pi = Math.round(frac * Math.PI);
  const pi2 = Math.round(frac * (Math.PI * Math.PI));
  console.log("/// PI << 56, (PI ** 2) << 56, (PI * 2) << 56, etc");
  console.log(`pub const PIS56: i64 = ${pi}i64;`);
  console.log(`pub const PIS56_SQUARE: i64 = ${pi2}i64;`);
}

populate();

